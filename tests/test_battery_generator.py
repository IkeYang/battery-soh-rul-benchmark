from pathlib import Path

import pandas as pd

from tools.generate_battery_benchmark import BenchmarkConfig, generate_benchmark


def test_generate_benchmark_creates_hidden_eval_labels(tmp_path: Path) -> None:
    config = BenchmarkConfig(
        output_dir=tmp_path,
        seed=123,
        dev_cells=6,
        train_cells=10,
        eval_cells=8,
        min_cycles=90,
        max_cycles=120,
    )

    manifest = generate_benchmark(config)

    assert manifest["dev_rows"] > 0
    assert manifest["train_rows"] > manifest["dev_rows"]
    assert manifest["eval_rows"] > manifest["dev_rows"]

    agent_start = tmp_path / "agent-start"
    work_files = {p.relative_to(agent_start).as_posix() for p in agent_start.rglob("*") if p.is_file()}
    assert "dev-data/dev_features.csv" in work_files
    assert "dev-data/dev_labels.csv" in work_files
    assert "dev-data/train/cycle_features.csv" in work_files
    assert "dev-data/train/cycle_labels.csv" in work_files
    assert "scorer/eval-data/eval_labels.csv" not in work_files

    sample_submission = pd.read_csv(agent_start / "dev-data" / "sample_submission.csv")
    dev_features = pd.read_csv(agent_start / "dev-data" / "dev_features.csv")
    assert sample_submission[["cell_id", "cycle_index"]].equals(dev_features[["cell_id", "cycle_index"]])

    eval_features = pd.read_csv(tmp_path / "scorer" / "eval-data" / "eval_features.csv")
    eval_labels = pd.read_csv(tmp_path / "scorer" / "eval-data" / "eval_labels.csv")
    assert len(eval_features) == len(eval_labels)
    assert set(eval_labels.columns) == {
        "cell_id",
        "cycle_index",
        "soh",
        "rul_cycles",
        "eol_cycle",
        "anomaly_type",
        "anomaly_severity",
    }
    assert eval_features["cell_id"].str.startswith("eval_").all()
    assert eval_features["cell_id"].nunique() == config.eval_cells


def test_generate_benchmark_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    kwargs = dict(seed=777, dev_cells=4, train_cells=5, eval_cells=4, min_cycles=80, max_cycles=95)

    generate_benchmark(BenchmarkConfig(output_dir=first, **kwargs))
    generate_benchmark(BenchmarkConfig(output_dir=second, **kwargs))

    first_eval = (first / "scorer" / "eval-data" / "eval_labels.csv").read_bytes()
    second_eval = (second / "scorer" / "eval-data" / "eval_labels.csv").read_bytes()
    assert first_eval == second_eval


def test_eval_split_hides_capacity_calibration_shift(tmp_path: Path) -> None:
    config = BenchmarkConfig(
        output_dir=tmp_path,
        seed=321,
        dev_cells=8,
        train_cells=12,
        eval_cells=9,
        min_cycles=100,
        max_cycles=130,
    )

    generate_benchmark(config)

    train_features = pd.read_csv(tmp_path / "agent-start/dev-data/train/cycle_features.csv")
    train_labels = pd.read_csv(tmp_path / "agent-start/dev-data/train/cycle_labels.csv")
    eval_features = pd.read_csv(tmp_path / "scorer/eval-data/eval_features.csv")
    eval_labels = pd.read_csv(tmp_path / "scorer/eval-data/eval_labels.csv")

    train = train_features.merge(train_labels, on=["cell_id", "cycle_index"], validate="one_to_one")
    hidden = eval_features.merge(eval_labels, on=["cell_id", "cycle_index"], validate="one_to_one")
    train_proxy_mae = ((train["measured_capacity_ah"] / train["nominal_capacity_ah"]) - train["soh"]).abs().mean()
    hidden_proxy_mae = ((hidden["measured_capacity_ah"] / hidden["nominal_capacity_ah"]) - hidden["soh"]).abs().mean()

    assert train_proxy_mae < 0.006
    assert hidden_proxy_mae > train_proxy_mae * 3.0


def test_eval_split_contains_out_of_distribution_lifetime_regimes(tmp_path: Path) -> None:
    config = BenchmarkConfig(
        output_dir=tmp_path,
        seed=654,
        dev_cells=8,
        train_cells=12,
        eval_cells=10,
        min_cycles=100,
        max_cycles=130,
    )

    generate_benchmark(config)

    train_meta = pd.read_json(tmp_path / "docs/cell_metadata.json")
    train_eol = train_meta[train_meta["split"] == "train"]["eol_cycle"]
    eval_eol = train_meta[train_meta["split"] == "eval"]["eol_cycle"]

    assert eval_eol.max() > train_eol.quantile(0.95) + 35
    assert eval_eol.min() < train_eol.quantile(0.05) - 20
