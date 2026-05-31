#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
import hashlib
import tarfile
from collections.abc import Iterator
from pathlib import Path


ROOT_NAME = "battery-soh-rul-benchmark"
WORK_EXCLUDES = {
    "scorer/eval-data/eval_labels.csv",
    "scorer/eval-data/baseline_metrics.json",
    "split_manifest.json",
    "docs/cell_metadata.json",
}
FIXED_MTIME = 0


def _iter_included_files(root: Path, include_hidden_eval: bool) -> list[tuple[Path, str]]:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(root).as_posix()
        if not include_hidden_eval and not rel.startswith("agent-start/"):
            continue
        if rel.startswith("raw-data/") or rel.startswith(".tmp/") or rel.startswith(".pytest_cache/") or rel.startswith("__pycache__/"):
            continue
        if "/__pycache__/" in rel or rel.endswith(".pyc"):
            continue
        if rel.startswith("baseline/") and not include_hidden_eval:
            continue
        if rel.startswith("tests/") and not include_hidden_eval:
            continue
        if rel.startswith("tools/") and not include_hidden_eval:
            continue
        if rel in WORK_EXCLUDES and not include_hidden_eval:
            continue
        if rel.startswith("scorer/eval-data/") and not include_hidden_eval:
            continue
        files.append((path, rel))
    return files


def _add_tree(tar: tarfile.TarFile, root: Path, include_hidden_eval: bool) -> None:
    for path, rel in _iter_included_files(root, include_hidden_eval):
        info = tar.gettarinfo(str(path), arcname=f"{ROOT_NAME}/{rel}")
        info.mtime = FIXED_MTIME
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        if info.isfile():
            info.mode = 0o644
        with path.open("rb") as fh:
            tar.addfile(info, fh)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_sha256sums(output_dir: Path, names: list[str]) -> None:
    lines = [f"{_sha256(output_dir / name)}  {name}" for name in names]
    (output_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


@contextmanager
def _open_reproducible_tar(path: Path) -> Iterator[tarfile.TarFile]:
    out = path.open("wb")
    gz = gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=FIXED_MTIME)
    tar = tarfile.open(fileobj=gz, mode="w")
    try:
        yield tar
    finally:
        tar.close()
        gz.close()
        out.close()


def make_assets(root: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    work_path = output_dir / "battery_work.tar.gz"
    judge_path = output_dir / "battery_judge.tar.gz"
    with _open_reproducible_tar(work_path) as tar:
        _add_tree(tar, root, include_hidden_eval=False)
    with _open_reproducible_tar(judge_path) as tar:
        _add_tree(tar, root, include_hidden_eval=True)
    _write_sha256sums(output_dir, ["battery_work.tar.gz", "battery_judge.tar.gz"])
    return {"work": str(work_path), "judge": str(judge_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "harness-assets" / "battery_soh_rul")
    args = parser.parse_args()
    print(make_assets(args.root, args.output_dir))


if __name__ == "__main__":
    main()
