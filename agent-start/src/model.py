from __future__ import annotations

from pathlib import Path

import pandas as pd
import joblib
import numpy as np


MODEL_FILE = "battery_model.joblib"
ANOMALY_TYPES = [
    "normal",
    "capacity_drop",
    "resistance_spike",
    "thermal_event",
    "sensor_drift",
    "recovery_relaxation",
]


def _align(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame.reindex(columns=columns, fill_value=0.0)


def train_model(features: pd.DataFrame, labels: pd.DataFrame, model_dir: Path) -> None:
    joined = features.merge(labels, on=["cell_id", "cycle_index"], how="inner")
    joined["cycle_bin"] = pd.cut(joined["cycle_index"], bins=[0, 100, 250, 500, 750, 2000], labels=False, include_lowest=True)
    group_cols = ["chemistry", "protocol_family", "cycle_bin"]
    grouped = (
        joined.groupby(group_cols, dropna=False)
        .agg(
            soh_mean=("soh", "mean"),
            rul_mean=("rul_cycles", "mean"),
            severity_mean=("anomaly_severity", "mean"),
            anomaly_mode=("anomaly_type", lambda s: s.mode().iloc[0] if not s.mode().empty else "normal"),
        )
        .reset_index()
    )
    fallback = {
        "soh_mean": float(joined["soh"].mean()),
        "rul_mean": float(joined["rul_cycles"].mean()),
        "severity_mean": float(joined["anomaly_severity"].mean()),
        "anomaly_mode": "normal",
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"grouped": grouped, "fallback": fallback}, model_dir / MODEL_FILE)


def predict(features: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    bundle = joblib.load(model_dir / MODEL_FILE)
    data = features.copy()
    data["cycle_bin"] = pd.cut(data["cycle_index"], bins=[0, 100, 250, 500, 750, 2000], labels=False, include_lowest=True)
    merged = data.merge(bundle["grouped"], on=["chemistry", "protocol_family", "cycle_bin"], how="left")
    fallback = bundle["fallback"]
    output = features[["cell_id", "cycle_index"]].copy()
    output["predicted_soh"] = np.clip(merged["soh_mean"].fillna(fallback["soh_mean"]).to_numpy(dtype=float), 0.50, 1.08)
    output["predicted_rul_cycles"] = np.clip(merged["rul_mean"].fillna(fallback["rul_mean"]).to_numpy(dtype=float), 0.0, 2000.0)
    output["predicted_anomaly_type"] = merged["anomaly_mode"].fillna(fallback["anomaly_mode"]).astype(str)
    output["predicted_anomaly_severity"] = np.clip(merged["severity_mean"].fillna(fallback["severity_mean"]).to_numpy(dtype=float), 0.0, 1.0)
    return output
