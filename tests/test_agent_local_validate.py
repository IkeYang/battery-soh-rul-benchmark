from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from tools.generate_battery_benchmark import BenchmarkConfig, generate_benchmark


def test_agent_local_validate_runs_on_visible_dev_split(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    generate_benchmark(BenchmarkConfig(output_dir=root, seed=911, train_cells=8, dev_cells=4, eval_cells=3, min_cycles=70, max_cycles=85))
    source_agent = Path(__file__).resolve().parents[1] / "agent-start"
    agent_copy = tmp_path / "agent-copy"
    shutil.copytree(source_agent, agent_copy, ignore=shutil.ignore_patterns("dev-data", "artifacts", "submission.csv", "__pycache__", "*.pyc"))
    shutil.copytree(root / "agent-start" / "dev-data", agent_copy / "dev-data")

    completed = subprocess.run(
        [
            sys.executable,
            "local_validate.py",
            "--train-dir",
            "dev-data/train",
            "--dev-features",
            "dev-data/dev_features.csv",
            "--dev-labels",
            "dev-data/dev_labels.csv",
            "--model-dir",
            "artifacts/local-validate",
            "--sample-train-cells",
            "5",
            "--sample-dev-cells",
            "3",
        ],
        cwd=agent_copy,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["valid"] is True
    assert payload["rows"] > 0
    assert payload["metrics"]["soh_mae"] >= 0.0
    assert payload["metrics"]["rul_mae"] >= 0.0
    assert 0.0 <= payload["metrics"]["anomaly_macro_f1"] <= 1.0
    assert 0.0 <= payload["local_score"] <= 100.0
    assert 0.0 <= payload["raw_quality"] <= 1.0
    assert 0.0 <= payload["stress_local_score"] <= 100.0
    assert 0.0 <= payload["stress_raw_quality"] <= 1.0
    assert 0.0 <= payload["regime_stress_local_score"] <= 100.0
    assert 0.0 <= payload["regime_stress_raw_quality"] <= 1.0
    assert 0.0 <= payload["challenge_local_score"] <= 100.0
    assert 0.0 <= payload["challenge_raw_quality"] <= 1.0
    assert payload["robust_raw_quality"] <= payload["raw_quality"]
    assert payload["robust_raw_quality"] <= payload["stress_raw_quality"]
    assert payload["robust_raw_quality"] <= payload["regime_stress_raw_quality"]
    assert payload["robust_raw_quality"] <= payload["challenge_raw_quality"]
    assert "anomaly_type_accuracy_on_anomaly" in payload["metrics"]
    assert "anomaly_type_accuracy_on_anomaly" in payload["stress_metrics"]
    assert "anomaly_type_accuracy_on_anomaly" in payload["regime_stress_metrics"]
    assert "anomaly_type_accuracy_on_anomaly" in payload["challenge_metrics"]
    assert payload["challenge_metrics"]["eol_mae"] >= 0.0
    assert payload["challenge_metrics"]["severity_rmse_on_anomaly"] >= 0.0


def test_agent_local_validate_can_run_from_project_root() -> None:
    root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "agent-start/local_validate.py",
            "--sample-train-cells",
            "5",
            "--sample-dev-cells",
            "3",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["valid"] is True
    assert payload["train_cells"] == 5
    assert payload["dev_cells"] == 3
    assert "stress_metrics" in payload
    assert "regime_stress_metrics" in payload
    assert "challenge_metrics" in payload
