#!/usr/bin/env python3
"""Generate a deterministic battery SOH/RUL benchmark dataset.

The generated data is a private derivative benchmark: public battery-aging
datasets inform field ranges, but the hidden labels come from deterministic
augmentation and fault injection that is not present in the public sources.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ANOMALY_TYPES = [
    "normal",
    "capacity_drop",
    "resistance_spike",
    "thermal_event",
    "sensor_drift",
    "recovery_relaxation",
]


@dataclass(frozen=True)
class BenchmarkConfig:
    output_dir: Path
    seed: int = 20260527
    train_cells: int = 900
    dev_cells: int = 180
    eval_cells: int = 520
    min_cycles: int = 420
    max_cycles: int = 980


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def _rng_for(master: np.random.Generator, split: str, index: int) -> np.random.Generator:
    salt = {"train": 1009, "dev": 2003, "eval": 3001}[split]
    return np.random.default_rng(int(master.integers(1, 2**31 - 1)) + salt + index * 7919)


def _sample_protocol(rng: np.random.Generator, split: str) -> dict:
    chemistries = ["NMC", "LFP", "NCA", "LMO"]
    chemistry = str(rng.choice(chemistries, p=[0.36, 0.30, 0.22, 0.12]))
    protocol_family = str(rng.choice(["fast_charge", "fleet_mixed", "cold_storage", "high_soc"], p=[0.32, 0.34, 0.16, 0.18]))
    base_capacity = float(rng.normal({"NMC": 2.9, "LFP": 3.15, "NCA": 2.75, "LMO": 2.45}[chemistry], 0.12))
    temp_base = float(rng.normal({"cold_storage": 17.0, "high_soc": 31.0}.get(protocol_family, 25.0), 4.2))
    charge_c = float(np.clip(rng.normal({"fast_charge": 2.4, "fleet_mixed": 1.25, "cold_storage": 0.85, "high_soc": 1.45}[protocol_family], 0.35), 0.35, 3.6))
    discharge_c = float(np.clip(rng.normal(1.0, 0.28), 0.25, 2.4))
    upper_soc = float(np.clip(rng.normal({"high_soc": 0.965}.get(protocol_family, 0.91), 0.025), 0.82, 0.995))
    lower_soc = float(np.clip(rng.normal(0.11, 0.025), 0.03, 0.20))
    eval_shift = 0.0 if split != "eval" else float(rng.normal(0.08, 0.04))
    return {
        "chemistry": chemistry,
        "protocol_family": protocol_family,
        "nominal_capacity_ah": max(1.8, base_capacity),
        "ambient_temp_c": temp_base,
        "charge_c_rate": charge_c + eval_shift,
        "discharge_c_rate": discharge_c,
        "upper_soc": upper_soc,
        "lower_soc": lower_soc,
    }


def _make_anomaly_plan(rng: np.random.Generator, total_cycles: int, split: str) -> list[dict]:
    count = int(rng.integers(2, 6 if split != "eval" else 8))
    plan: list[dict] = []
    for _ in range(count):
        kind = str(rng.choice(ANOMALY_TYPES[1:], p=[0.24, 0.24, 0.19, 0.20, 0.13]))
        start = int(rng.integers(max(12, total_cycles // 12), max(20, total_cycles - 28)))
        duration = int(rng.integers(3, 28))
        severity = float(np.clip(rng.beta(2.2, 3.0) * 1.35, 0.08, 0.95))
        plan.append({"type": kind, "start": start, "end": min(total_cycles, start + duration), "severity": severity})
    return plan


def _eval_lifetime_offset(index: int, rng: np.random.Generator) -> int:
    """Create hidden-only lifetime regimes that are not represented in train/dev."""
    bucket = index % 5
    if bucket == 0:
        return int(rng.integers(120, 210))
    if bucket == 1:
        return -int(rng.integers(80, 155))
    return int(rng.integers(10, 80))


def _eval_capacity_calibration(cycles: np.ndarray, index: int, rng: np.random.Generator) -> np.ndarray:
    """Hidden evaluation has realistic sensor calibration drift not present in visible labels."""
    t = cycles / max(float(cycles[-1]), 1.0)
    direction = -1.0 if index % 2 else 1.0
    offset = direction * float(rng.uniform(0.010, 0.024))
    slope = -direction * float(rng.uniform(0.004, 0.016))
    curvature = float(rng.uniform(-0.010, 0.010))
    return offset + slope * t + curvature * (t - 0.5) ** 2


def _generate_cell(split: str, index: int, master_rng: np.random.Generator, config: BenchmarkConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = _rng_for(master_rng, split, index)
    protocol = _sample_protocol(rng, split)
    total_cycles = int(rng.integers(config.min_cycles, config.max_cycles + 1))
    cycles = np.arange(1, total_cycles + 1, dtype=int)
    eol_cycle = int(total_cycles + rng.integers(30, 180))
    if split == "eval":
        eol_cycle = max(total_cycles + 5, eol_cycle + _eval_lifetime_offset(index, rng))
    knee = float(np.clip(rng.normal(0.66, 0.10), 0.42, 0.86))
    early_rate = float(rng.uniform(0.030, 0.105))
    late_rate = float(rng.uniform(0.080, 0.240))
    if split == "eval":
        regime = index % 4
        if regime == 0:
            knee = float(np.clip(knee - rng.uniform(0.16, 0.26), 0.30, 0.72))
            late_rate *= float(rng.uniform(1.35, 1.85))
        elif regime == 1:
            knee = float(np.clip(knee + rng.uniform(0.12, 0.24), 0.50, 0.92))
            early_rate *= float(rng.uniform(0.60, 0.86))
        elif regime == 2:
            early_rate *= float(rng.uniform(1.20, 1.55))
            late_rate *= float(rng.uniform(0.70, 1.05))
    t = cycles / max(eol_cycle, 1)
    soh_clean = 1.0 - early_rate * t - late_rate * np.maximum(0.0, t - knee) ** 1.65
    soh_clean -= 0.010 * _sigmoid((t - knee) * 30.0)
    soh_noise = rng.normal(0.0, 0.0025, size=len(cycles)).cumsum() * 0.015 + rng.normal(0.0, 0.003, size=len(cycles))
    soh = soh_clean + soh_noise

    anomaly_type = np.array(["normal"] * len(cycles), dtype=object)
    anomaly_severity = np.zeros(len(cycles), dtype=float)
    internal_resistance = 0.030 + 0.025 * t + rng.normal(0.0, 0.0015, len(cycles))
    max_temp = protocol["ambient_temp_c"] + 4.0 + 2.4 * protocol["charge_c_rate"] + 5.0 * t + rng.normal(0.0, 1.1, len(cycles))
    coulombic_eff = 0.995 - 0.008 * t + rng.normal(0.0, 0.002, len(cycles))
    charge_time_min = 62.0 / protocol["charge_c_rate"] * (1.0 + 0.08 * t) + rng.normal(0.0, 2.2, len(cycles))
    voltage_mean = 3.61 + 0.08 * (protocol["upper_soc"] - protocol["lower_soc"]) - 0.05 * t + rng.normal(0.0, 0.010, len(cycles))
    voltage_std = 0.030 + 0.025 * t + rng.normal(0.0, 0.004, len(cycles))

    for anomaly in _make_anomaly_plan(rng, total_cycles, split):
        start = anomaly["start"]
        end = anomaly["end"]
        severity = anomaly["severity"]
        sl = slice(start - 1, end - 1)
        local = np.linspace(0.2, 1.0, max(0, end - start))
        if len(local) == 0:
            continue
        anomaly_type[sl] = anomaly["type"]
        anomaly_severity[sl] = np.maximum(anomaly_severity[sl], severity)
        if anomaly["type"] == "capacity_drop":
            soh[sl] -= severity * (0.020 + 0.030 * local)
        elif anomaly["type"] == "resistance_spike":
            internal_resistance[sl] += severity * (0.015 + 0.020 * local)
            coulombic_eff[sl] -= severity * 0.010
        elif anomaly["type"] == "thermal_event":
            max_temp[sl] += severity * (8.0 + 12.0 * local)
            voltage_std[sl] += severity * 0.018
        elif anomaly["type"] == "sensor_drift":
            voltage_mean[sl] += severity * np.linspace(0.010, 0.055, len(local))
            charge_time_min[sl] += severity * np.linspace(2.0, 11.0, len(local))
        elif anomaly["type"] == "recovery_relaxation":
            soh[sl] += severity * (0.006 + 0.014 * np.sin(np.linspace(0, math.pi, len(local))))
            internal_resistance[sl] -= severity * 0.006

    soh = np.clip(soh, 0.55, 1.05)
    capacity_calibration = 0.0
    if split == "eval":
        capacity_calibration = _eval_capacity_calibration(cycles, index, rng)
    observed_capacity = protocol["nominal_capacity_ah"] * (soh + capacity_calibration) + rng.normal(0.0, 0.010, len(cycles))
    rul_cycles = np.maximum(eol_cycle - cycles, 0)
    cell_id = f"{split}_{index:04d}"
    capacity_roll3 = pd.Series(observed_capacity).rolling(3, min_periods=1).mean().to_numpy()
    resistance_delta = np.diff(np.r_[internal_resistance[0], internal_resistance])
    capacity_delta = np.diff(np.r_[observed_capacity[0], observed_capacity])

    features = pd.DataFrame(
        {
            "cell_id": cell_id,
            "cycle_index": cycles,
            "chemistry": protocol["chemistry"],
            "protocol_family": protocol["protocol_family"],
            "nominal_capacity_ah": protocol["nominal_capacity_ah"],
            "charge_c_rate": protocol["charge_c_rate"] + rng.normal(0.0, 0.025, len(cycles)),
            "discharge_c_rate": protocol["discharge_c_rate"] + rng.normal(0.0, 0.020, len(cycles)),
            "ambient_temp_c": protocol["ambient_temp_c"] + rng.normal(0.0, 1.4, len(cycles)),
            "upper_soc": protocol["upper_soc"],
            "lower_soc": protocol["lower_soc"],
            "measured_capacity_ah": observed_capacity,
            "capacity_roll3_ah": capacity_roll3,
            "capacity_delta_ah": capacity_delta,
            "internal_resistance_ohm": internal_resistance,
            "resistance_delta_ohm": resistance_delta,
            "max_temp_c": max_temp,
            "coulombic_efficiency": coulombic_eff,
            "charge_time_min": charge_time_min,
            "voltage_mean_v": voltage_mean,
            "voltage_std_v": voltage_std,
        }
    )
    labels = pd.DataFrame(
        {
            "cell_id": cell_id,
            "cycle_index": cycles,
            "soh": soh,
            "rul_cycles": rul_cycles,
            "eol_cycle": eol_cycle,
            "anomaly_type": anomaly_type,
            "anomaly_severity": anomaly_severity,
        }
    )
    meta = {
        "cell_id": cell_id,
        "split": split,
        "cycles": int(total_cycles),
        "eol_cycle": int(eol_cycle),
        **protocol,
        "anomaly_rate": float((anomaly_type != "normal").mean()),
        "hidden_eval_capacity_calibration_mae": float(np.mean(np.abs(capacity_calibration))) if split == "eval" else 0.0,
    }
    return features, labels, meta


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _baseline_metrics(eval_labels: pd.DataFrame) -> dict:
    normal_rate = float((eval_labels["anomaly_type"] == "normal").mean())
    return {
        "label_distribution": eval_labels["anomaly_type"].value_counts().to_dict(),
        "normal_rate": normal_rate,
        "weights": {
            "soh_score": 0.24,
            "rul_score": 0.24,
            "eol_score": 0.10,
            "knee_score": 0.10,
            "anomaly_f1_score": 0.18,
            "anomaly_type_score": 0.09,
            "severity_score": 0.05,
        },
        "score_anchors": [
            {"name": "weak_baseline", "raw_quality": 0.18, "score": 0.0},
            {"name": "simple_global_model", "raw_quality": 0.36, "score": 5.0},
            {"name": "starter_grouped_statistics", "raw_quality": 0.42, "score": 6.5},
            {"name": "ordinary_hgb_reference", "raw_quality": 0.5805711178213633, "score": 9.78041991750218},
            {"name": "ordinary_hgb_multitask", "raw_quality": 0.6892158840109565, "score": 12.0},
            {"name": "observed_iter6_30m", "raw_quality": 0.7013097249484304, "score": 13.0},
            {"name": "observed_iter6_1h", "raw_quality": 0.7193325395358704, "score": 14.0},
            {"name": "short_run_guardrail", "raw_quality": 0.7353059943553691, "score": 15.0},
            {"name": "iter9_thirty_minute_guardrail", "raw_quality": 0.7433049347998876, "score": 15.0},
            {"name": "iter8_one_hour_guardrail", "raw_quality": 0.7615917042309768, "score": 28.0},
            {"name": "iter11_two_hour_guardrail", "raw_quality": 0.7626304050926715, "score": 28.5},
            {"name": "two_hour_acceptance_ceiling", "raw_quality": 0.7735352840883063, "score": 30.0},
            {"name": "eight_hour_target_floor", "raw_quality": 0.7752919808318431, "score": 42.0},
            {"name": "eight_hour_target_mid", "raw_quality": 0.79000, "score": 45.0},
            {"name": "eight_hour_target_ceiling", "raw_quality": 0.80500, "score": 50.0},
            {"name": "advanced_ensemble", "raw_quality": 0.84, "score": 70.0},
            {"name": "expert_target", "raw_quality": 0.92, "score": 100.0},
        ],
    }


def generate_benchmark(config: BenchmarkConfig) -> dict:
    output_dir = Path(config.output_dir)
    if output_dir.exists():
        for child in ["agent-start/dev-data", "scorer/eval-data", "split_manifest.json"]:
            target = output_dir / child
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
    master_rng = np.random.default_rng(config.seed)
    split_counts = {"train": config.train_cells, "dev": config.dev_cells, "eval": config.eval_cells}
    frames: dict[str, list[pd.DataFrame]] = {key: [] for key in split_counts}
    label_frames: dict[str, list[pd.DataFrame]] = {key: [] for key in split_counts}
    meta: list[dict] = []
    for split, count in split_counts.items():
        for idx in range(count):
            features, labels, cell_meta = _generate_cell(split, idx, master_rng, config)
            frames[split].append(features)
            label_frames[split].append(labels)
            meta.append(cell_meta)

    train_features = pd.concat(frames["train"], ignore_index=True)
    train_labels = pd.concat(label_frames["train"], ignore_index=True)
    dev_features = pd.concat(frames["dev"], ignore_index=True)
    dev_labels = pd.concat(label_frames["dev"], ignore_index=True)
    eval_features = pd.concat(frames["eval"], ignore_index=True)
    eval_labels = pd.concat(label_frames["eval"], ignore_index=True)

    agent_dir = output_dir / "agent-start"
    scorer_dir = output_dir / "scorer" / "eval-data"
    _write_csv(agent_dir / "dev-data" / "train" / "cycle_features.csv", train_features)
    _write_csv(agent_dir / "dev-data" / "train" / "cycle_labels.csv", train_labels)
    _write_csv(agent_dir / "dev-data" / "dev_features.csv", dev_features)
    _write_csv(agent_dir / "dev-data" / "dev_labels.csv", dev_labels)
    dev_sample = dev_features[["cell_id", "cycle_index"]].copy()
    dev_sample["predicted_soh"] = 0.90
    dev_sample["predicted_rul_cycles"] = 300.0
    dev_sample["predicted_anomaly_type"] = "normal"
    dev_sample["predicted_anomaly_severity"] = 0.0
    eval_sample = eval_features[["cell_id", "cycle_index"]].copy()
    eval_sample["predicted_soh"] = 0.90
    eval_sample["predicted_rul_cycles"] = 300.0
    eval_sample["predicted_anomaly_type"] = "normal"
    eval_sample["predicted_anomaly_severity"] = 0.0
    _write_csv(agent_dir / "dev-data" / "sample_submission.csv", dev_sample)
    _write_csv(scorer_dir / "eval_features.csv", eval_features)
    _write_csv(scorer_dir / "eval_labels.csv", eval_labels)
    _write_csv(scorer_dir / "sample_submission.csv", eval_sample)
    metrics = _baseline_metrics(eval_labels)
    (scorer_dir / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=True), encoding="utf-8")

    manifest = {
        "seed": config.seed,
        "train_cells": config.train_cells,
        "dev_cells": config.dev_cells,
        "eval_cells": config.eval_cells,
        "train_rows": int(len(train_features)),
        "dev_rows": int(len(dev_features)),
        "eval_rows": int(len(eval_features)),
        "splits": {k: [m["cell_id"] for m in meta if m["split"] == k] for k in split_counts},
        "source_note": "Private deterministic derivative dataset calibrated from public Li-ion battery aging ranges; hidden labels are generated by this benchmark.",
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    (output_dir / "docs").mkdir(parents=True, exist_ok=True)
    (output_dir / "docs" / "cell_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--train-cells", type=int, default=900)
    parser.add_argument("--dev-cells", type=int, default=180)
    parser.add_argument("--eval-cells", type=int, default=520)
    parser.add_argument("--min-cycles", type=int, default=420)
    parser.add_argument("--max-cycles", type=int, default=980)
    args = parser.parse_args()
    manifest = generate_benchmark(
        BenchmarkConfig(
            output_dir=args.output_dir,
            seed=args.seed,
            train_cells=args.train_cells,
            dev_cells=args.dev_cells,
            eval_cells=args.eval_cells,
            min_cycles=args.min_cycles,
            max_cycles=args.max_cycles,
        )
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
