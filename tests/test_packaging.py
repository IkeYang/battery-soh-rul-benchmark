import tarfile
import json
import hashlib
import os
from pathlib import Path

from tools.generate_battery_benchmark import BenchmarkConfig, generate_benchmark
from tools.package_assets import make_assets


def test_work_asset_contains_only_agent_start_and_no_hidden_labels(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    generate_benchmark(BenchmarkConfig(output_dir=root, seed=901, train_cells=4, dev_cells=2, eval_cells=3, min_cycles=70, max_cycles=85))
    assets = make_assets(root, tmp_path)

    with tarfile.open(assets["work"], "r:gz") as tar:
        names = tar.getnames()

    assert names
    assert all(name.startswith("battery-soh-rul-benchmark/agent-start/") for name in names)
    assert not any("eval_labels.csv" in name for name in names)
    assert not any("baseline_metrics.json" in name for name in names)
    assert not any("split_manifest.json" in name for name in names)
    assert not any("/scorer/" in name for name in names)


def test_judge_asset_contains_eval_labels_and_scorer(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    generate_benchmark(BenchmarkConfig(output_dir=root, seed=902, train_cells=4, dev_cells=2, eval_cells=3, min_cycles=70, max_cycles=85))
    scorer_dir = root / "scorer"
    scorer_dir.mkdir(exist_ok=True)
    (scorer_dir / "evaluate.py").write_text("print('placeholder')\n", encoding="utf-8")
    assets = make_assets(root, tmp_path)

    with tarfile.open(assets["judge"], "r:gz") as tar:
        names = set(tar.getnames())

    assert "battery-soh-rul-benchmark/scorer/eval-data/eval_labels.csv" in names
    assert "battery-soh-rul-benchmark/scorer/evaluate.py" in names


def test_harness_uses_score_sum_parser_for_sebench_020_compatibility() -> None:
    harness_path = Path(__file__).resolve().parents[1] / "harness" / "battery_soh_rul_anomaly.json"
    payload = json.loads(harness_path.read_text(encoding="utf-8"))

    assert payload["judge"]["parser"] == "score_sum"


def test_harness_pip_install_uses_china_friendly_mirror() -> None:
    harness_path = Path(__file__).resolve().parents[1] / "harness" / "battery_soh_rul_anomaly.json"
    payload = json.loads(harness_path.read_text(encoding="utf-8"))

    setup_cmds = payload["work"]["setup_cmds"] + payload["judge"]["setup_cmds"]
    pip_commands = [cmd for cmd in setup_cmds if "pip install" in cmd]

    assert pip_commands
    assert all("SEBENCH_PYPI_INDEX_URL" in cmd for cmd in pip_commands)
    assert all("pypi.tuna.tsinghua.edu.cn/simple" in cmd for cmd in pip_commands)


def test_asset_packaging_is_reproducible(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    generate_benchmark(BenchmarkConfig(output_dir=root, seed=903, train_cells=4, dev_cells=2, eval_cells=3, min_cycles=70, max_cycles=85))

    first = make_assets(root, tmp_path / "first")
    for path in root.rglob("*"):
        if path.is_file():
            os.utime(path, (1_800_000_000, 1_800_000_000))
    second = make_assets(root, tmp_path / "second")

    for key in ("work", "judge"):
        first_hash = hashlib.sha256(Path(first[key]).read_bytes()).hexdigest()
        second_hash = hashlib.sha256(Path(second[key]).read_bytes()).hexdigest()
        assert first_hash == second_hash


def test_asset_packaging_writes_matching_sha256sums(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    output_dir = tmp_path / "assets"
    generate_benchmark(BenchmarkConfig(output_dir=root, seed=910, train_cells=4, dev_cells=2, eval_cells=3, min_cycles=70, max_cycles=85))

    make_assets(root, output_dir)

    sums_path = output_dir / "SHA256SUMS"
    assert sums_path.exists()
    expected = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        expected[name] = digest
    assert expected["battery_work.tar.gz"] == hashlib.sha256((output_dir / "battery_work.tar.gz").read_bytes()).hexdigest()
    assert expected["battery_judge.tar.gz"] == hashlib.sha256((output_dir / "battery_judge.tar.gz").read_bytes()).hexdigest()
