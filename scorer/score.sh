#!/usr/bin/env bash
set -euo pipefail

AGENT_DIR="${1:-agent-start}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    PYTHON_BIN=python
  fi
fi

PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" "$ROOT/scorer/run_eval.py" \
  --agent-dir "$ROOT/$AGENT_DIR" \
  --features "$ROOT/scorer/eval-data/eval_features.csv" \
  --labels "$ROOT/scorer/eval-data/eval_labels.csv" \
  --metrics "$ROOT/scorer/eval-data/baseline_metrics.json"
