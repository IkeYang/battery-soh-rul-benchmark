#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-start"))

from src.features import make_features  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", type=Path, default=ROOT / "agent-start/dev-data/train/cycle_features.csv")
    parser.add_argument("--train-labels", type=Path, default=ROOT / "agent-start/dev-data/train/cycle_labels.csv")
    parser.add_argument("--input", type=Path, default=ROOT / "scorer/eval-data/eval_features.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "baseline/reference_hgb_submission.csv")
    args = parser.parse_args()

    train_features = pd.read_csv(args.train_features)
    train_labels = pd.read_csv(args.train_labels)
    eval_features = pd.read_csv(args.input)
    x_train = make_features(train_features)
    x_eval = make_features(eval_features).reindex(columns=x_train.columns, fill_value=0.0)

    common = dict(max_iter=260, learning_rate=0.045, max_leaf_nodes=63, l2_regularization=0.02)
    soh = HistGradientBoostingRegressor(**common, random_state=101).fit(x_train, train_labels["soh"])
    rul = HistGradientBoostingRegressor(**common, random_state=102).fit(x_train, train_labels["rul_cycles"])
    severity = HistGradientBoostingRegressor(max_iter=220, learning_rate=0.055, max_leaf_nodes=63, l2_regularization=0.03, random_state=103).fit(
        x_train, train_labels["anomaly_severity"]
    )
    anomaly = HistGradientBoostingClassifier(max_iter=180, learning_rate=0.055, max_leaf_nodes=63, l2_regularization=0.03, random_state=104).fit(
        x_train, train_labels["anomaly_type"].astype(str)
    )

    output = eval_features[["cell_id", "cycle_index"]].copy()
    output["predicted_soh"] = np.clip(soh.predict(x_eval), 0.50, 1.08)
    output["predicted_rul_cycles"] = np.clip(rul.predict(x_eval), 0.0, 2000.0)
    output["predicted_anomaly_type"] = anomaly.predict(x_eval)
    output["predicted_anomaly_severity"] = np.clip(severity.predict(x_eval), 0.0, 1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"wrote {len(output)} rows to {args.output}")


if __name__ == "__main__":
    main()
