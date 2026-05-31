import shutil
from pathlib import Path

import json

from scorer.run_eval import format_score_sum_lines
from scorer.run_eval import run_evaluation
from tools.generate_battery_benchmark import BenchmarkConfig, generate_benchmark


def test_score_sum_lines_encode_score_as_pass_rate_and_total_score() -> None:
    lines = format_score_sum_lines({"score": 13.941056417386184})

    assert lines[-2] == "TOTAL_SCORE 13.941056417386184"
    assert lines[-1] == "CASES_TOTAL 10000"
    assert sum(" OK " in line for line in lines if line.startswith("CASE ")) == 1394
    assert sum(" WA " in line for line in lines if line.startswith("CASE ")) == 8606


def test_run_evaluation_returns_invalid_json_for_broken_agent(tmp_path: Path) -> None:
    generate_benchmark(BenchmarkConfig(output_dir=tmp_path, seed=321, train_cells=4, dev_cells=2, eval_cells=3, min_cycles=70, max_cycles=85))
    source_agent = Path(__file__).resolve().parents[1] / "agent-start"
    agent_copy = tmp_path / "agent-copy"
    shutil.copytree(source_agent, agent_copy, ignore=shutil.ignore_patterns("dev-data", "artifacts", "submission.csv"))
    shutil.copytree(tmp_path / "agent-start" / "dev-data", agent_copy / "dev-data")
    (agent_copy / "train.py").write_text("raise RuntimeError('broken training')\n", encoding="utf-8")

    result = run_evaluation(
        agent_dir=agent_copy,
        features_path=tmp_path / "scorer" / "eval-data" / "eval_features.csv",
        labels_path=tmp_path / "scorer" / "eval-data" / "eval_labels.csv",
        metrics_path=tmp_path / "scorer" / "eval-data" / "baseline_metrics.json",
        timeout_train=20,
        timeout_predict=20,
    )

    assert result["valid"] is False
    assert result["score"] == 0.0
    assert "train failed" in result["summary"]
    json.dumps(result)


def test_run_evaluation_uses_current_python_when_python_is_not_on_path(tmp_path: Path, monkeypatch) -> None:
    generate_benchmark(BenchmarkConfig(output_dir=tmp_path, seed=322, train_cells=4, dev_cells=2, eval_cells=3, min_cycles=70, max_cycles=85))
    source_agent = Path(__file__).resolve().parents[1] / "agent-start"
    agent_copy = tmp_path / "agent-copy"
    shutil.copytree(source_agent, agent_copy, ignore=shutil.ignore_patterns("dev-data", "artifacts", "submission.csv"))
    shutil.copytree(tmp_path / "agent-start" / "dev-data", agent_copy / "dev-data")
    (agent_copy / "train.py").write_text("raise RuntimeError('broken training')\n", encoding="utf-8")
    monkeypatch.setenv("PATH", "")

    result = run_evaluation(
        agent_dir=agent_copy,
        features_path=tmp_path / "scorer" / "eval-data" / "eval_features.csv",
        labels_path=tmp_path / "scorer" / "eval-data" / "eval_labels.csv",
        metrics_path=tmp_path / "scorer" / "eval-data" / "baseline_metrics.json",
        timeout_train=20,
        timeout_predict=20,
    )

    assert result["valid"] is False
    assert result["score"] == 0.0
    assert "train failed" in result["summary"]
