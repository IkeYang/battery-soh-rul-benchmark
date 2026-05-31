#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.model import predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    features = pd.read_csv(args.input)
    submission = predict(features, args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"wrote {len(submission)} predictions to {args.output}")


if __name__ == "__main__":
    main()
