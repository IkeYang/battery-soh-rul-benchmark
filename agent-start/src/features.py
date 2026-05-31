from __future__ import annotations

import pandas as pd


CATEGORICAL_COLUMNS = ["chemistry", "protocol_family"]


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create numeric model features from cycle-level battery telemetry."""
    data = frame.copy()
    data["cycle_fraction_proxy"] = data["cycle_index"] / data.groupby("cell_id")["cycle_index"].transform("max").clip(lower=1)
    data["dod"] = data["upper_soc"] - data["lower_soc"]
    data["temp_x_charge"] = data["ambient_temp_c"] * data["charge_c_rate"]
    data["resistance_per_capacity"] = data["internal_resistance_ohm"] / data["measured_capacity_ah"].clip(lower=0.1)
    numeric = data.drop(columns=["cell_id"], errors="ignore")
    numeric = pd.get_dummies(numeric, columns=[col for col in CATEGORICAL_COLUMNS if col in numeric.columns], dummy_na=True)
    return numeric.apply(pd.to_numeric, errors="coerce").fillna(0.0)
