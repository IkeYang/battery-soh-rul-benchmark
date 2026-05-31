#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.model import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("dev-data/train"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    features = pd.read_csv(args.train_dir / "cycle_features.csv")
    labels = pd.read_csv(args.train_dir / "cycle_labels.csv")
    train_model(features, labels, args.model_dir)
    print(f"trained battery SOH/RUL model on {len(features)} rows")


if __name__ == "__main__":
    main()
