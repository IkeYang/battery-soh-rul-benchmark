#!/usr/bin/env python3
"""Evaluate battery SOH/RUL and anomalous-cycle submissions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "cell_id",
    "cycle_index",
    "predicted_soh",
    "predicted_rul_cycles",
    "predicted_anomaly_type",
    "predicted_anomaly_severity",
]

ANOMALY_TYPES = [
    "normal",
    "capacity_drop",
    "resistance_spike",
    "thermal_event",
    "sensor_drift",
    "recovery_relaxation",
]


def _invalid(summary: str) -> dict:
    return {
        "valid": False,
        "score": 0.0,
        "pass_rate": 0.0,
        "total_score": 0.0,
        "summary": summary,
        "metrics": {},
        "details": [{"name": "format", "status": "FAILED", "message": summary}],
    }


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted))) if len(actual) else 0.0


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2))) if len(actual) else 0.0


def _clip01(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def _macro_f1(actual: np.ndarray, predicted: np.ndarray, labels: list[str]) -> float:
    scores = []
    for label in labels:
        tp = float(((actual == label) & (predicted == label)).sum())
        fp = float(((actual != label) & (predicted == label)).sum())
        fn = float(((actual == label) & (predicted != label)).sum())
        if tp + fp + fn == 0:
            continue
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else 0.0


def _score_from_anchors(raw_quality: float, metrics: dict) -> float:
    anchors = metrics.get("score_anchors") or []
    clean = []
    for item in anchors:
        try:
            clean.append({"raw_quality": float(item["raw_quality"]), "score": float(item["score"])})
        except (KeyError, TypeError, ValueError):
            continue
    if not clean:
        return float(np.clip(raw_quality, 0.0, 1.0) * 100.0)
    clean.sort(key=lambda item: item["raw_quality"])
    if raw_quality <= clean[0]["raw_quality"]:
        return 0.0
    if raw_quality >= clean[-1]["raw_quality"]:
        return clean[-1]["score"]
    for left, right in zip(clean, clean[1:]):
        if left["raw_quality"] <= raw_quality <= right["raw_quality"]:
            span = max(right["raw_quality"] - left["raw_quality"], 1e-9)
            ratio = (raw_quality - left["raw_quality"]) / span
            return float(left["score"] + ratio * (right["score"] - left["score"]))
    return 0.0


def _load_metrics(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _validate(features: pd.DataFrame, labels: pd.DataFrame, submission: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    missing = [col for col in REQUIRED_COLUMNS if col not in submission.columns]
    if missing:
        return None, f"missing required columns: {missing}"
    if len(submission) != len(features):
        return None, f"row count mismatch: expected {len(features)}, got {len(submission)}"
    sub = submission[REQUIRED_COLUMNS].copy()
    sub["cycle_index"] = pd.to_numeric(sub["cycle_index"], errors="coerce").astype("Int64")
    if sub["cycle_index"].isna().any():
        return None, "cycle_index contains missing or non-integer values"
    if sub.duplicated(["cell_id", "cycle_index"]).any():
        return None, "duplicate cell_id-cycle_index rows in submission"
    expected = features[["cell_id", "cycle_index"]].copy()
    expected["cycle_index"] = expected["cycle_index"].astype("Int64")
    aligned = expected.merge(sub, on=["cell_id", "cycle_index"], how="left", validate="one_to_one")
    if aligned["predicted_soh"].isna().any():
        return None, "submission keys do not match eval features"
    for col in ["predicted_soh", "predicted_rul_cycles", "predicted_anomaly_severity"]:
        aligned[col] = pd.to_numeric(aligned[col], errors="coerce")
        if aligned[col].isna().any() or np.isinf(aligned[col].to_numpy()).any():
            return None, f"{col} contains non-numeric, missing, or infinite values"
    bad_types = sorted(set(aligned["predicted_anomaly_type"].astype(str)) - set(ANOMALY_TYPES))
    if bad_types:
        return None, f"unknown anomaly types: {bad_types[:5]}"
    if len(labels) != len(features):
        return None, "internal scorer error: features and labels length mismatch"
    return aligned, None


def _knee_error(labels: pd.DataFrame, submission: pd.DataFrame) -> float:
    errors = []
    merged = labels[["cell_id", "cycle_index", "soh"]].copy()
    merged["predicted_soh"] = submission["predicted_soh"].to_numpy(dtype=float)
    for _cell_id, group in merged.groupby("cell_id", sort=False):
        truth_idx = int(group.iloc[(group["soh"] - 0.90).abs().argsort().iloc[0]]["cycle_index"])
        pred_idx = int(group.iloc[(group["predicted_soh"] - 0.90).abs().argsort().iloc[0]]["cycle_index"])
        errors.append(abs(truth_idx - pred_idx))
    return float(np.mean(errors)) if errors else 0.0


def evaluate_submission(
    features_path: Path | str,
    labels_path: Path | str,
    submission_path: Path | str,
    metrics_path: Path | str | None = None,
) -> dict:
    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)
    try:
        submission = pd.read_csv(submission_path)
    except Exception as exc:
        return _invalid(f"could not read submission: {type(exc).__name__}: {exc}")
    aligned, error = _validate(features, labels, submission)
    if error:
        return _invalid(error)
    assert aligned is not None
    metrics_cfg = _load_metrics(Path(metrics_path) if metrics_path is not None else None)

    actual_soh = labels["soh"].to_numpy(dtype=float)
    pred_soh = np.clip(aligned["predicted_soh"].to_numpy(dtype=float), 0.45, 1.10)
    actual_rul = labels["rul_cycles"].to_numpy(dtype=float)
    pred_rul = np.clip(aligned["predicted_rul_cycles"].to_numpy(dtype=float), 0.0, 2000.0)
    actual_severity = labels["anomaly_severity"].to_numpy(dtype=float)
    pred_severity = _clip01(aligned["predicted_anomaly_severity"])
    actual_type = labels["anomaly_type"].astype(str).to_numpy()
    pred_type = aligned["predicted_anomaly_type"].astype(str).to_numpy()

    anomaly_mask = actual_type != "normal"
    early_mask = features["cycle_index"].to_numpy(dtype=float) <= features.groupby("cell_id")["cycle_index"].transform("max").to_numpy(dtype=float) * 0.35
    knee_mask = (actual_soh <= 0.93) & (actual_soh >= 0.84)

    soh_mae = _mae(actual_soh, pred_soh)
    soh_knee_mae = _mae(actual_soh[knee_mask], pred_soh[knee_mask]) if knee_mask.any() else soh_mae
    rul_mae = _mae(actual_rul, pred_rul)
    early_rul_mae = _mae(actual_rul[early_mask], pred_rul[early_mask]) if early_mask.any() else rul_mae
    eol_pred = aligned.groupby("cell_id").apply(lambda g: float(np.median(g["cycle_index"] + g["predicted_rul_cycles"])))
    eol_true = labels.groupby("cell_id")["eol_cycle"].first()
    eol_mae = _mae(eol_true.loc[eol_pred.index].to_numpy(dtype=float), eol_pred.to_numpy(dtype=float))
    knee_mae = _knee_error(labels, aligned)
    anomaly_macro_f1 = _macro_f1(actual_type, pred_type, ANOMALY_TYPES)
    anomaly_binary_f1 = _macro_f1(np.where(anomaly_mask, "abnormal", "normal"), np.where(pred_type != "normal", "abnormal", "normal"), ["normal", "abnormal"])
    type_accuracy_on_anomaly = float((actual_type[anomaly_mask] == pred_type[anomaly_mask]).mean()) if anomaly_mask.any() else 1.0
    severity_rmse = _rmse(actual_severity[anomaly_mask], pred_severity[anomaly_mask]) if anomaly_mask.any() else 0.0

    soh_score = math.exp(-soh_mae / 0.020) * 0.70 + math.exp(-soh_knee_mae / 0.024) * 0.30
    rul_score = math.exp(-rul_mae / 145.0) * 0.55 + math.exp(-early_rul_mae / 190.0) * 0.45
    eol_score = math.exp(-eol_mae / 120.0)
    knee_score = math.exp(-knee_mae / 95.0)
    anomaly_f1_score = 0.55 * anomaly_macro_f1 + 0.45 * anomaly_binary_f1
    anomaly_type_score = type_accuracy_on_anomaly
    severity_score = math.exp(-severity_rmse / 0.22)
    component_scores = {
        "soh_score": float(soh_score),
        "rul_score": float(rul_score),
        "eol_score": float(eol_score),
        "knee_score": float(knee_score),
        "anomaly_f1_score": float(anomaly_f1_score),
        "anomaly_type_score": float(anomaly_type_score),
        "severity_score": float(severity_score),
    }
    weights = metrics_cfg.get("weights") or {
        "soh_score": 0.24,
        "rul_score": 0.24,
        "eol_score": 0.10,
        "knee_score": 0.10,
        "anomaly_f1_score": 0.18,
        "anomaly_type_score": 0.09,
        "severity_score": 0.05,
    }
    raw_quality = float(sum(component_scores[name] * float(weights.get(name, 0.0)) for name in component_scores) / max(sum(float(v) for v in weights.values()), 1e-9))
    score = max(0.0, min(100.0, _score_from_anchors(raw_quality, metrics_cfg)))
    result_metrics = {
        "soh_mae": float(soh_mae),
        "soh_knee_mae": float(soh_knee_mae),
        "rul_mae": float(rul_mae),
        "early_rul_mae": float(early_rul_mae),
        "eol_mae": float(eol_mae),
        "knee_cycle_mae": float(knee_mae),
        "anomaly_macro_f1": float(anomaly_macro_f1),
        "anomaly_binary_f1": float(anomaly_binary_f1),
        "anomaly_type_accuracy_on_anomaly": float(type_accuracy_on_anomaly),
        "severity_rmse_on_anomaly": float(severity_rmse),
        "raw_quality": raw_quality,
        **component_scores,
    }
    return {
        "valid": True,
        "score": score,
        "pass_rate": score / 100.0,
        "total_score": score,
        "summary": f"score={score:.3f}, raw_quality={raw_quality:.4f}, soh_mae={soh_mae:.5f}, rul_mae={rul_mae:.2f}, anomaly_macro_f1={anomaly_macro_f1:.3f}",
        "metrics": result_metrics,
        "details": [
            {"name": "soh", "status": "PASSED", "message": f"MAE {soh_mae:.5f}, knee MAE {soh_knee_mae:.5f}"},
            {"name": "rul", "status": "PASSED", "message": f"MAE {rul_mae:.2f}, early MAE {early_rul_mae:.2f}"},
            {"name": "anomaly", "status": "PASSED", "message": f"macro F1 {anomaly_macro_f1:.3f}, type accuracy {type_accuracy_on_anomaly:.3f}"},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()
    result = evaluate_submission(args.features, args.labels, args.submission, args.metrics)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
