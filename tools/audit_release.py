#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.release_hygiene import scan_release_hygiene


HIDDEN_PATTERNS = [
    "/scorer/",
    "/docs/",
    "/harness/",
    "/baseline/",
    "/raw-data/",
    "/eval_labels.csv",
    "/baseline_metrics.json",
    "/split_manifest.json",
    "/cell_metadata.json",
    "/task.yaml",
]
REQUIRED_JUDGE_FILES = [
    "battery-soh-rul-benchmark/scorer/eval-data/eval_labels.csv",
    "battery-soh-rul-benchmark/scorer/eval-data/baseline_metrics.json",
    "battery-soh-rul-benchmark/scorer/evaluate.py",
    "battery-soh-rul-benchmark/scorer/run_eval.py",
    "battery-soh-rul-benchmark/scorer/score.sh",
]
JUDGE_TREE_MATCH_FILES = [
    "split_manifest.json",
    "baseline/starter-score-log.json",
    "baseline/reference-hgb-score-log.json",
]


def _check(ok: bool, message: str, **extra: Any) -> dict[str, Any]:
    return {"ok": bool(ok), "message": message, **extra}


def _tar_names(path: Path) -> tuple[list[str], str | None]:
    try:
        with tarfile.open(path, "r:gz") as tar:
            return tar.getnames(), None
    except (tarfile.TarError, EOFError, OSError) as exc:
        return [], f"{type(exc).__name__}: {exc}"


def _read_tar_text(path: Path, member_name: str) -> tuple[str | None, str | None]:
    try:
        with tarfile.open(path, "r:gz") as tar:
            member = tar.getmember(member_name)
            fh = tar.extractfile(member)
            if fh is None:
                return None, f"{member_name} is not a regular file"
            with io.TextIOWrapper(fh, encoding="utf-8") as text_fh:
                return text_fh.read(), None
    except KeyError:
        return None, f"{member_name} not found"
    except (tarfile.TarError, EOFError, OSError, UnicodeDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_expected_hashes(asset_dir: Path) -> dict[str, str]:
    path = asset_dir / "SHA256SUMS"
    expected: dict[str, str] = {}
    if not path.exists():
        return expected
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            expected[parts[1]] = parts[0]
    return expected


def audit_release(
    root: Path,
    asset_dir: Path,
    min_train_rows: int = 300_000,
    min_dev_rows: int = 50_000,
    min_eval_rows: int = 150_000,
) -> dict[str, Any]:
    root = Path(root)
    asset_dir = Path(asset_dir)
    work_path = asset_dir / "battery_work.tar.gz"
    judge_path = asset_dir / "battery_judge.tar.gz"
    checks: dict[str, dict[str, Any]] = {}

    expected_hashes = _read_expected_hashes(asset_dir)
    actual_hashes = {
        "battery_work.tar.gz": _sha256(work_path) if work_path.exists() else "",
        "battery_judge.tar.gz": _sha256(judge_path) if judge_path.exists() else "",
    }
    checks["asset_hashes"] = _check(
        bool(expected_hashes) and all(expected_hashes.get(name) == digest for name, digest in actual_hashes.items()),
        "asset hashes match SHA256SUMS" if expected_hashes else "missing SHA256SUMS",
        expected=expected_hashes,
        actual=actual_hashes,
    )

    work_names, work_error = _tar_names(work_path) if work_path.exists() else ([], "missing asset")
    checks["work_asset_readable"] = _check(
        work_error is None,
        "work asset is a readable tar.gz" if work_error is None else "work asset is not readable",
        error=work_error,
    )
    leaked = sorted(name for name in work_names for pattern in HIDDEN_PATTERNS if pattern in f"/{name}")
    checks["work_asset_no_hidden_files"] = _check(
        work_error is None
        and not leaked
        and all(name.startswith("battery-soh-rul-benchmark/agent-start/") for name in work_names),
        "work asset contains only agent-start files" if work_error is None else "work asset could not be inspected",
        leaked=leaked[:20],
        files=len(work_names),
    )

    judge_names_list, judge_error = _tar_names(judge_path) if judge_path.exists() else ([], "missing asset")
    checks["judge_asset_readable"] = _check(
        judge_error is None,
        "judge asset is a readable tar.gz" if judge_error is None else "judge asset is not readable",
        error=judge_error,
    )
    judge_names = set(judge_names_list)
    missing_judge = [name for name in REQUIRED_JUDGE_FILES if name not in judge_names]
    checks["judge_asset_required_files"] = _check(
        judge_error is None and not missing_judge,
        "judge asset contains required scorer and hidden eval files" if judge_error is None else "judge asset could not be inspected",
        missing=missing_judge,
    )
    stale_files = []
    tree_match_errors = []
    if judge_error is None:
        for rel in JUDGE_TREE_MATCH_FILES:
            source_path = root / rel
            asset_member = f"battery-soh-rul-benchmark/{rel}"
            asset_text, asset_read_error = _read_tar_text(judge_path, asset_member)
            if asset_read_error:
                if source_path.exists() or rel == "split_manifest.json":
                    tree_match_errors.append(asset_read_error)
                continue
            if not source_path.exists():
                continue
            if asset_text != source_path.read_text(encoding="utf-8"):
                stale_files.append(rel)
    checks["judge_asset_matches_current_tree"] = _check(
        judge_error is None and not tree_match_errors and not stale_files,
        "judge asset manifest and calibration files match current tree" if judge_error is None else "judge asset could not be inspected",
        stale_files=stale_files,
        errors=tree_match_errors,
    )

    train_features = pd.read_csv(root / "agent-start/dev-data/train/cycle_features.csv")
    dev_features = pd.read_csv(root / "agent-start/dev-data/dev_features.csv")
    eval_features = pd.read_csv(root / "scorer/eval-data/eval_features.csv")
    scale_ok = len(train_features) >= min_train_rows and len(dev_features) >= min_dev_rows and len(eval_features) >= min_eval_rows
    checks["dataset_scale"] = _check(
        scale_ok,
        "dataset row counts meet configured minimums",
        train_rows=len(train_features),
        dev_rows=len(dev_features),
        eval_rows=len(eval_features),
        train_cells=int(train_features["cell_id"].nunique()),
        dev_cells=int(dev_features["cell_id"].nunique()),
        eval_cells=int(eval_features["cell_id"].nunique()),
        minimums={"train_rows": min_train_rows, "dev_rows": min_dev_rows, "eval_rows": min_eval_rows},
    )

    harness = json.loads((root / "harness/battery_soh_rul_anomaly.json").read_text(encoding="utf-8"))
    checks["harness_parser"] = _check(
        harness.get("judge", {}).get("parser") == "score_sum",
        "harness parser is score_sum for SE-Bench 0.2.0 compatibility",
        parser=harness.get("judge", {}).get("parser"),
    )

    hygiene = scan_release_hygiene(root, asset_dir)
    checks["release_hygiene"] = _check(
        hygiene["ok"],
        "release tree and work asset passed hygiene scan" if hygiene["ok"] else "release hygiene scan found issues",
        issues=hygiene["issues"][:20],
    )

    result = {"ok": all(item["ok"] for item in checks.values()), "checks": checks}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--asset-dir", type=Path, default=Path(__file__).resolve().parents[2] / "harness-assets" / "battery_soh_rul")
    parser.add_argument("--min-train-rows", type=int, default=300_000)
    parser.add_argument("--min-dev-rows", type=int, default=50_000)
    parser.add_argument("--min-eval-rows", type=int, default=150_000)
    args = parser.parse_args()
    result = audit_release(args.root, args.asset_dir, args.min_train_rows, args.min_dev_rows, args.min_eval_rows)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
