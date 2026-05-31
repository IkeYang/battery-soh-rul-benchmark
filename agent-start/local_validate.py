#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.model import predict, train_model


ANOMALY_TYPES = [
    "normal",
    "capacity_drop",
    "resistance_spike",
    "thermal_event",
    "sensor_drift",
    "recovery_relaxation",
]


def _resolve_path(path: Path) -> Path:
    if path.exists() or path.is_absolute():
        return path
    agent_root = Path(__file__).resolve().parent
    candidate = agent_root / path
    return candidate if candidate.exists() else path


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
        scores.append(0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else 0.0


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted))) if len(actual) else 0.0


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2))) if len(actual) else 0.0


def _knee_error(labels: pd.DataFrame, submission: pd.DataFrame) -> float:
    errors = []
    merged = labels[["cell_id", "cycle_index", "soh"]].copy()
    merged["predicted_soh"] = submission["predicted_soh"].to_numpy(dtype=float)
    for _cell_id, group in merged.groupby("cell_id", sort=False):
        truth_idx = int(group.iloc[(group["soh"] - 0.90).abs().argsort().iloc[0]]["cycle_index"])
        pred_idx = int(group.iloc[(group["predicted_soh"] - 0.90).abs().argsort().iloc[0]]["cycle_index"])
        errors.append(abs(truth_idx - pred_idx))
    return float(np.mean(errors)) if errors else 0.0


def _sample_cells(frame: pd.DataFrame, max_cells: int | None) -> pd.DataFrame:
    if max_cells is None or max_cells <= 0:
        return frame
    cells = sorted(frame["cell_id"].drop_duplicates().astype(str).tolist())[:max_cells]
    return frame[frame["cell_id"].astype(str).isin(cells)].reset_index(drop=True)


def _make_eval_like_stress_features(features: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic sensor/protocol shifts while keeping dev labels fixed."""
    shifted = features.copy().sort_values(["cell_id", "cycle_index"]).reset_index(drop=True)
    cell_order = {cell_id: i for i, cell_id in enumerate(sorted(shifted["cell_id"].astype(str).unique()))}
    cell_idx = shifted["cell_id"].astype(str).map(cell_order).fillna(0).to_numpy(dtype=float)
    max_cycle = shifted.groupby("cell_id", sort=False)["cycle_index"].transform("max").clip(lower=1).to_numpy(dtype=float)
    t = shifted["cycle_index"].to_numpy(dtype=float) / max_cycle
    direction = np.where((cell_idx.astype(int) % 2) == 0, 1.0, -1.0)
    offset = direction * (0.010 + 0.002 * (cell_idx % 5.0))
    slope = -direction * (0.006 + 0.001 * (cell_idx % 7.0))
    curvature = 0.004 * np.sin(cell_idx)
    capacity_drift = offset + slope * t + curvature * (t - 0.5) ** 2
    nominal = shifted["nominal_capacity_ah"].clip(lower=0.1).to_numpy(dtype=float)
    shifted["measured_capacity_ah"] = shifted["measured_capacity_ah"].to_numpy(dtype=float) + nominal * capacity_drift
    shifted["capacity_roll3_ah"] = shifted.groupby("cell_id", sort=False)["measured_capacity_ah"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    shifted["capacity_delta_ah"] = shifted.groupby("cell_id", sort=False)["measured_capacity_ah"].diff().fillna(0.0)
    shifted["charge_c_rate"] = shifted["charge_c_rate"].to_numpy(dtype=float) + 0.06 + 0.02 * np.sin(cell_idx)
    shifted["ambient_temp_c"] = shifted["ambient_temp_c"].to_numpy(dtype=float) + 1.5 * np.cos(cell_idx % 11.0)
    return shifted.reindex(index=features.index)


def _make_regime_stress_split(features: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a harder dev proxy with eval-like lifetime regimes and capacity calibration.

    This is still derived only from visible dev labels. It deliberately changes
    the target EOL/RUL relationship so local validation penalizes solutions that
    memorize the visible split's lifetime range or trust observed capacity ratio
    without cell-level calibration.
    """
    shifted = _make_eval_like_stress_features(features).copy().sort_values(["cell_id", "cycle_index"]).reset_index(drop=True)
    shifted_labels = labels.copy().sort_values(["cell_id", "cycle_index"]).reset_index(drop=True)
    cell_order = {cell_id: i for i, cell_id in enumerate(sorted(shifted["cell_id"].astype(str).unique()))}
    cell_idx = shifted["cell_id"].astype(str).map(cell_order).fillna(0).to_numpy(dtype=int)
    max_cycle = shifted.groupby("cell_id", sort=False)["cycle_index"].transform("max").clip(lower=1).to_numpy(dtype=float)
    t = shifted["cycle_index"].to_numpy(dtype=float) / max_cycle

    early = (cell_idx % 4) == 0
    late = (cell_idx % 4) == 1
    mixed = (cell_idx % 4) == 2
    soh_delta = np.zeros(len(shifted), dtype=float)
    soh_delta[early] = -0.010 * t[early] - 0.018 * np.maximum(0.0, t[early] - 0.46) ** 1.5
    soh_delta[late] = 0.008 * t[late] + 0.010 * np.maximum(0.0, t[late] - 0.64) ** 1.4
    soh_delta[mixed] = -0.006 * np.sin(np.pi * t[mixed]) + 0.004 * t[mixed]
    shifted_labels["soh"] = np.clip(shifted_labels["soh"].to_numpy(dtype=float) + soh_delta, 0.50, 1.08)

    per_cell_delta: dict[str, int] = {}
    for cell_id, ordinal in cell_order.items():
        if ordinal % 5 == 0:
            per_cell_delta[cell_id] = 120 + 9 * (ordinal % 11)
        elif ordinal % 5 == 1:
            per_cell_delta[cell_id] = -(70 + 7 * (ordinal % 13))
        else:
            per_cell_delta[cell_id] = 25 + 5 * (ordinal % 9)
    eol_delta = shifted["cell_id"].astype(str).map(per_cell_delta).to_numpy(dtype=float)
    max_by_cell = shifted.groupby("cell_id", sort=False)["cycle_index"].transform("max").to_numpy(dtype=float)
    eol_new = np.maximum(shifted_labels["eol_cycle"].to_numpy(dtype=float) + eol_delta, max_by_cell + 5.0)
    shifted_labels["eol_cycle"] = eol_new
    shifted_labels["rul_cycles"] = np.maximum(eol_new - shifted["cycle_index"].to_numpy(dtype=float), 0.0)

    type_values = shifted_labels["anomaly_type"].astype(str).to_numpy()
    severity = shifted_labels["anomaly_severity"].to_numpy(dtype=float)
    sensor_candidate = (cell_idx % 6 == 2) & (t > 0.18) & (t < 0.34)
    recovery_candidate = (cell_idx % 7 == 3) & (t > 0.58) & (t < 0.72)
    type_values = np.where(sensor_candidate & (type_values == "normal"), "sensor_drift", type_values)
    type_values = np.where(recovery_candidate & (type_values == "normal"), "recovery_relaxation", type_values)
    severity = np.where(sensor_candidate | recovery_candidate, np.maximum(severity, 0.22 + 0.18 * np.sin(np.pi * t) ** 2), severity)
    shifted_labels["anomaly_type"] = type_values
    shifted_labels["anomaly_severity"] = np.clip(severity, 0.0, 1.0)

    return shifted.reindex(index=features.index), shifted_labels.reindex(index=labels.index)


def _make_challenge_split(features: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hidden-like visible proxy emphasizing the remaining hard cases.

    The hidden split has wider EOL-gap variance, capacity-calibration drift,
    more abnormal cycles, and several early/late knee regimes. This transform
    reproduces those mechanisms from visible labels without using hidden data.
    """
    shifted = features.copy().sort_values(["cell_id", "cycle_index"]).reset_index(drop=True)
    shifted_labels = labels.copy().sort_values(["cell_id", "cycle_index"]).reset_index(drop=True)
    cell_order = {cell_id: i for i, cell_id in enumerate(sorted(shifted["cell_id"].astype(str).unique()))}
    cell_idx = shifted["cell_id"].astype(str).map(cell_order).fillna(0).to_numpy(dtype=int)
    max_cycle = shifted.groupby("cell_id", sort=False)["cycle_index"].transform("max").clip(lower=1).to_numpy(dtype=float)
    cycle = shifted["cycle_index"].to_numpy(dtype=float)
    t_obs = cycle / max_cycle

    direction = np.where((cell_idx % 2) == 0, 1.0, -1.0)
    offset = direction * (0.012 + 0.0025 * (cell_idx % 5))
    slope = -direction * (0.006 + 0.0015 * (cell_idx % 7))
    curvature = ((cell_idx % 3) - 1.0) * 0.004
    calibration = offset + slope * t_obs + curvature * (t_obs - 0.5) ** 2
    nominal = shifted["nominal_capacity_ah"].clip(lower=0.1).to_numpy(dtype=float)
    shifted["measured_capacity_ah"] = shifted["measured_capacity_ah"].to_numpy(dtype=float) + nominal * calibration
    shifted["capacity_roll3_ah"] = shifted.groupby("cell_id", sort=False)["measured_capacity_ah"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    shifted["capacity_delta_ah"] = shifted.groupby("cell_id", sort=False)["measured_capacity_ah"].diff().fillna(0.0)
    shifted["charge_c_rate"] = shifted["charge_c_rate"].to_numpy(dtype=float) + 0.08 + 0.025 * np.sin(cell_idx)
    shifted["ambient_temp_c"] = shifted["ambient_temp_c"].to_numpy(dtype=float) + 1.2 * np.cos(cell_idx % 9.0)

    eol_delta_by_cell: dict[str, int] = {}
    for cell_id, ordinal in cell_order.items():
        bucket = ordinal % 5
        if bucket == 0:
            eol_delta_by_cell[cell_id] = 145 + 13 * (ordinal % 11)
        elif bucket == 1:
            eol_delta_by_cell[cell_id] = -(85 + 7 * (ordinal % 10))
        else:
            eol_delta_by_cell[cell_id] = 35 + 8 * (ordinal % 8)
    eol_delta = shifted["cell_id"].astype(str).map(eol_delta_by_cell).to_numpy(dtype=float)
    eol_new = shifted_labels["eol_cycle"].to_numpy(dtype=float) + eol_delta
    eol_new = np.maximum(eol_new, max_cycle + 5.0)

    regime = cell_idx % 4
    t_eol = cycle / np.maximum(eol_new, 1.0)
    knee_shift = np.zeros(len(shifted), dtype=float)
    knee_shift[regime == 0] = -0.010 * t_eol[regime == 0] - 0.026 * np.maximum(0.0, t_obs[regime == 0] - 0.45) ** 1.55
    knee_shift[regime == 1] = 0.010 * t_eol[regime == 1] + 0.014 * np.maximum(0.0, t_obs[regime == 1] - 0.62) ** 1.35
    knee_shift[regime == 2] = -0.008 * np.sin(np.pi * t_obs[regime == 2]) + 0.004 * t_obs[regime == 2]
    shifted_labels["soh"] = np.clip(shifted_labels["soh"].to_numpy(dtype=float) + knee_shift, 0.50, 1.08)
    shifted_labels["eol_cycle"] = eol_new
    shifted_labels["rul_cycles"] = np.maximum(eol_new - cycle, 0.0)

    type_values = shifted_labels["anomaly_type"].astype(str).to_numpy()
    severity = shifted_labels["anomaly_severity"].to_numpy(dtype=float)
    cap_drop = (cell_idx % 8 == 0) & (t_obs > 0.48) & (t_obs < 0.58)
    res_spike = (cell_idx % 8 == 1) & (t_obs > 0.30) & (t_obs < 0.40)
    thermal = (cell_idx % 8 == 2) & (t_obs > 0.62) & (t_obs < 0.70)
    sensor = (cell_idx % 8 == 3) & (t_obs > 0.18) & (t_obs < 0.32)
    recovery = (cell_idx % 8 == 4) & (t_obs > 0.55) & (t_obs < 0.67)
    normal = type_values == "normal"
    type_values = np.where(cap_drop & normal, "capacity_drop", type_values)
    type_values = np.where(res_spike & normal, "resistance_spike", type_values)
    type_values = np.where(thermal & normal, "thermal_event", type_values)
    type_values = np.where(sensor & normal, "sensor_drift", type_values)
    type_values = np.where(recovery & normal, "recovery_relaxation", type_values)
    injected = cap_drop | res_spike | thermal | sensor | recovery
    severity = np.where(injected, np.maximum(severity, 0.24 + 0.34 * np.sin(np.pi * t_obs) ** 2), severity)
    shifted_labels["anomaly_type"] = type_values
    shifted_labels["anomaly_severity"] = np.clip(severity, 0.0, 1.0)

    return shifted.reindex(index=features.index), shifted_labels.reindex(index=labels.index)


def _evaluate(features: pd.DataFrame, labels: pd.DataFrame, submission: pd.DataFrame) -> dict:
    actual_soh = labels["soh"].to_numpy(dtype=float)
    pred_soh = np.clip(submission["predicted_soh"].to_numpy(dtype=float), 0.45, 1.10)
    actual_rul = labels["rul_cycles"].to_numpy(dtype=float)
    pred_rul = np.clip(submission["predicted_rul_cycles"].to_numpy(dtype=float), 0.0, 2000.0)
    actual_type = labels["anomaly_type"].astype(str).to_numpy()
    pred_type = submission["predicted_anomaly_type"].astype(str).to_numpy()
    anomaly_mask = actual_type != "normal"
    early_mask = features["cycle_index"].to_numpy(dtype=float) <= features.groupby("cell_id")["cycle_index"].transform("max").to_numpy(dtype=float) * 0.35
    knee_mask = (actual_soh <= 0.93) & (actual_soh >= 0.84)
    eol_pred = submission.groupby("cell_id").apply(lambda g: float(np.median(g["cycle_index"] + g["predicted_rul_cycles"])))
    eol_true = labels.groupby("cell_id")["eol_cycle"].first()
    return {
        "soh_mae": _mae(actual_soh, pred_soh),
        "soh_knee_mae": _mae(actual_soh[knee_mask], pred_soh[knee_mask]) if knee_mask.any() else _mae(actual_soh, pred_soh),
        "rul_mae": _mae(actual_rul, pred_rul),
        "early_rul_mae": _mae(actual_rul[early_mask], pred_rul[early_mask]) if early_mask.any() else _mae(actual_rul, pred_rul),
        "eol_mae": _mae(eol_true.loc[eol_pred.index].to_numpy(dtype=float), eol_pred.to_numpy(dtype=float)),
        "knee_cycle_mae": _knee_error(labels, submission),
        "anomaly_macro_f1": _macro_f1(actual_type, pred_type, ANOMALY_TYPES),
        "anomaly_binary_f1": _macro_f1(
            np.where(anomaly_mask, "abnormal", "normal"),
            np.where(pred_type != "normal", "abnormal", "normal"),
            ["normal", "abnormal"],
        ),
        "anomaly_type_accuracy_on_anomaly": float((actual_type[anomaly_mask] == pred_type[anomaly_mask]).mean())
        if anomaly_mask.any()
        else 1.0,
        "severity_rmse_on_anomaly": _rmse(
            labels.loc[anomaly_mask, "anomaly_severity"].to_numpy(dtype=float),
            np.clip(submission.loc[anomaly_mask, "predicted_anomaly_severity"].to_numpy(dtype=float), 0.0, 1.0),
        )
        if anomaly_mask.any()
        else 0.0,
    }


def _local_quality(metrics: dict) -> tuple[float, float]:
    soh_score = math.exp(-metrics["soh_mae"] / 0.020) * 0.70 + math.exp(-metrics["soh_knee_mae"] / 0.024) * 0.30
    rul_score = math.exp(-metrics["rul_mae"] / 145.0) * 0.55 + math.exp(-metrics["early_rul_mae"] / 190.0) * 0.45
    eol_score = math.exp(-metrics["eol_mae"] / 120.0)
    knee_score = math.exp(-metrics["knee_cycle_mae"] / 95.0)
    anomaly_f1_score = 0.55 * metrics["anomaly_macro_f1"] + 0.45 * metrics["anomaly_binary_f1"]
    anomaly_type_score = metrics["anomaly_type_accuracy_on_anomaly"]
    severity_score = math.exp(-metrics["severity_rmse_on_anomaly"] / 0.22)
    raw_quality = (
        0.24 * soh_score
        + 0.24 * rul_score
        + 0.10 * eol_score
        + 0.10 * knee_score
        + 0.18 * anomaly_f1_score
        + 0.09 * anomaly_type_score
        + 0.05 * severity_score
    )
    local_score = max(0.0, min(100.0, raw_quality * 100.0))
    return float(raw_quality), float(local_score)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("dev-data/train"))
    parser.add_argument("--dev-features", type=Path, default=Path("dev-data/dev_features.csv"))
    parser.add_argument("--dev-labels", type=Path, default=Path("dev-data/dev_labels.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/local-validate"))
    parser.add_argument("--sample-train-cells", type=int)
    parser.add_argument("--sample-dev-cells", type=int)
    parser.add_argument("--skip-stress", action="store_true")
    args = parser.parse_args()

    train_dir = _resolve_path(args.train_dir)
    dev_features_path = _resolve_path(args.dev_features)
    dev_labels_path = _resolve_path(args.dev_labels)
    train_features = pd.read_csv(train_dir / "cycle_features.csv")
    train_labels = pd.read_csv(train_dir / "cycle_labels.csv")
    dev_features = pd.read_csv(dev_features_path)
    dev_labels = pd.read_csv(dev_labels_path)
    train_features = _sample_cells(train_features, args.sample_train_cells)
    train_labels = train_labels.merge(train_features[["cell_id", "cycle_index"]], on=["cell_id", "cycle_index"], how="inner")
    dev_features = _sample_cells(dev_features, args.sample_dev_cells)
    dev_labels = dev_labels.merge(dev_features[["cell_id", "cycle_index"]], on=["cell_id", "cycle_index"], how="inner")

    if args.model_dir.exists():
        shutil.rmtree(args.model_dir)
    with tempfile.TemporaryDirectory(prefix="battery_local_validate_") as tmp:
        model_dir = args.model_dir if args.model_dir else Path(tmp) / "model"
        train_model(train_features, train_labels, model_dir)
        submission = predict(dev_features, model_dir)
    metrics = _evaluate(dev_features, dev_labels, submission)
    raw_quality, local_score = _local_quality(metrics)
    stress_metrics = None
    stress_raw_quality = None
    stress_local_score = None
    regime_stress_metrics = None
    regime_stress_raw_quality = None
    regime_stress_local_score = None
    challenge_metrics = None
    challenge_raw_quality = None
    challenge_local_score = None
    if not args.skip_stress:
        stress_features = _make_eval_like_stress_features(dev_features)
        stress_submission = predict(stress_features, model_dir)
        stress_metrics = _evaluate(stress_features, dev_labels, stress_submission)
        stress_raw_quality, stress_local_score = _local_quality(stress_metrics)
        regime_stress_features, regime_stress_labels = _make_regime_stress_split(dev_features, dev_labels)
        regime_stress_submission = predict(regime_stress_features, model_dir)
        regime_stress_metrics = _evaluate(regime_stress_features, regime_stress_labels, regime_stress_submission)
        regime_stress_raw_quality, regime_stress_local_score = _local_quality(regime_stress_metrics)
        challenge_features, challenge_labels = _make_challenge_split(dev_features, dev_labels)
        challenge_submission = predict(challenge_features, model_dir)
        challenge_metrics = _evaluate(challenge_features, challenge_labels, challenge_submission)
        challenge_raw_quality, challenge_local_score = _local_quality(challenge_metrics)
    quality_values = [raw_quality]
    if stress_raw_quality is not None:
        quality_values.append(stress_raw_quality)
    if regime_stress_raw_quality is not None:
        quality_values.append(regime_stress_raw_quality)
    if challenge_raw_quality is not None:
        quality_values.append(challenge_raw_quality)
    print(
        json.dumps(
            {
                "valid": True,
                "rows": int(len(dev_features)),
                "train_cells": int(train_features["cell_id"].nunique()),
                "dev_cells": int(dev_features["cell_id"].nunique()),
                "raw_quality": raw_quality,
                "local_score": local_score,
                "stress_raw_quality": stress_raw_quality,
                "stress_local_score": stress_local_score,
                "regime_stress_raw_quality": regime_stress_raw_quality,
                "regime_stress_local_score": regime_stress_local_score,
                "challenge_raw_quality": challenge_raw_quality,
                "challenge_local_score": challenge_local_score,
                "robust_raw_quality": min(quality_values),
                "visible_to_stress_drop": raw_quality - stress_raw_quality if stress_raw_quality is not None else 0.0,
                "visible_to_regime_stress_drop": raw_quality - regime_stress_raw_quality
                if regime_stress_raw_quality is not None
                else 0.0,
                "visible_to_challenge_drop": raw_quality - challenge_raw_quality if challenge_raw_quality is not None else 0.0,
                "metrics": metrics,
                "stress_metrics": stress_metrics,
                "regime_stress_metrics": regime_stress_metrics,
                "challenge_metrics": challenge_metrics,
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
