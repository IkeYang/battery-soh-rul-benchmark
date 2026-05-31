#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - yaml is available in the target env, fallback keeps CLI usable.
    yaml = None


REQUIRED_DOCS = [
    "docs/data_audit.md",
    "docs/workload_and_authenticity.md",
    "docs/pollution_control.md",
    "docs/pre_submission_review.md",
    "docs/environment_setup_cn.md",
    "docs/design_and_validation_plan.md",
]
REQUIRED_LONG_RUN_STAGES = ["30m", "1h", "2h", "8h"]
REMOTE_ASSET_DIR = Path("/root/sebench-assets/battery_soh_rul")


def resolve_asset_dir(default: Path, candidates: list[Path] | None = None) -> Path:
    all_candidates = [default, *(candidates or [])]
    for candidate in all_candidates:
        if (candidate / "SHA256SUMS").exists():
            return candidate
    return default


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text) or {}
    result: dict[str, Any] = {}
    for line in text.splitlines():
        if ":" in line and not line.startswith(" "):
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip().strip("\"'")
    return result


def _read_sha256sums(asset_dir: Path) -> dict[str, str]:
    path = asset_dir / "SHA256SUMS"
    if not path.exists():
        return {}
    hashes: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            hashes[parts[1]] = parts[0]
    return hashes


def _extract_score_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    match = re.search(r">>>> Start Structured Result\s*(\{.*?\})\s*>>>>> End Structured Result", text, re.S)
    if match:
        text = match.group(1)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return {"score": float(data["score"])} if "score" in data else data


def _doc_entries(root: Path) -> list[dict[str, Any]]:
    entries = []
    for rel in REQUIRED_DOCS:
        path = root / rel
        entries.append({"path": rel, "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0})
    return entries


def _load_long_run_evidence(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"completed": False, "runs": {}, "missing": list(REQUIRED_LONG_RUN_STAGES)}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"completed": False, "runs": {}, "missing": list(REQUIRED_LONG_RUN_STAGES), "error": "invalid JSON"}
    runs = data.get("runs", {}) if isinstance(data, dict) else {}
    missing = [stage for stage in REQUIRED_LONG_RUN_STAGES if not runs.get(stage)]
    completed = not missing and all(str(runs[stage].get("status", "")).lower() == "pass" for stage in REQUIRED_LONG_RUN_STAGES)
    return {"completed": completed, "runs": runs, "missing": missing}


def _load_remote_preflight(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"ok": False, "missing": True}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"ok": False, "missing": False, "error": "invalid JSON"}


def build_submission_manifest(
    root: Path,
    asset_dir: Path,
    long_run_evidence_path: Path | None = None,
    remote_preflight_path: Path | None = None,
) -> dict[str, Any]:
    root = Path(root)
    asset_dir = Path(asset_dir)
    task_yaml = _load_yaml(root / "task.yaml")
    harness_path = root / "harness/battery_soh_rul_anomaly.json"
    harness = json.loads(harness_path.read_text(encoding="utf-8")) if harness_path.exists() else {}
    hashes = _read_sha256sums(asset_dir)
    assets = {}
    for name in ["battery_work.tar.gz", "battery_judge.tar.gz"]:
        path = asset_dir / name
        assets[name] = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "sha256": hashes.get(name, ""),
        }

    docs = _doc_entries(root)
    long_runs = _load_long_run_evidence(long_run_evidence_path or root / "docs/long_run_evidence.json")
    remote_preflight = _load_remote_preflight(remote_preflight_path or root / "docs/remote_preflight.json")
    blocking_items = []
    if not all(item["exists"] for item in docs):
        blocking_items.append("required documentation is incomplete")
    if not all(asset["exists"] and asset["sha256"] for asset in assets.values()):
        blocking_items.append("asset package or SHA256SUMS is incomplete")
    if not long_runs["completed"]:
        blocking_items.append("30m/1h/2h/8h agent validation is not complete")
    if not remote_preflight.get("ok", False):
        blocking_items.append("remote live auth preflight is not passing")

    closed_loop_verified = not blocking_items
    return {
        "task": {
            "name": task_yaml.get("name", "battery-soh-rul-anomaly"),
            "title": task_yaml.get("title", ""),
            "category": task_yaml.get("category", ""),
            "agent_type": task_yaml.get("agent_type", ""),
            "task_id": harness.get("task_id", "battery_soh_rul_anomaly"),
            "network_required": task_yaml.get("network_required", False),
            "gpu_required": task_yaml.get("gpu_required", False),
        },
        "assets": assets,
        "baselines": {
            "starter": _extract_score_json(root / "baseline/starter-score-log.json"),
            "reference_hgb": _extract_score_json(root / "baseline/reference-hgb-score-log.json"),
        },
        "required_docs": docs,
        "validation": {
            "local_tests": "python -m pytest tests -q",
            "release_audit": "python tools/audit_release.py",
            "remote_live_preflight": "python tools/long_run_preflight.py --require-live-auth",
            "remote_preflight": remote_preflight,
            "long_runs": long_runs,
        },
        "status": {
            "closed_loop_verified": closed_loop_verified,
            "final_submission_ready": closed_loop_verified,
            "blocking_items": blocking_items,
        },
    }


def _to_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Battery SOH/RUL Submission Manifest",
        "",
        f"- Task ID: `{manifest['task']['task_id']}`",
        f"- Title: {manifest['task']['title']}",
        f"- Category: {manifest['task']['category']}",
        f"- GPU required: `{manifest['task']['gpu_required']}`",
        f"- Network required: `{manifest['task']['network_required']}`",
        f"- Final submission ready: `{manifest['status']['final_submission_ready']}`",
        "",
        "## Assets",
    ]
    for name, asset in manifest["assets"].items():
        lines.append(f"- `{name}`: exists={asset['exists']}, size={asset['size_bytes']}, sha256=`{asset['sha256']}`")
    lines.extend(["", "## Baselines"])
    for name, data in manifest["baselines"].items():
        lines.append(f"- `{name}`: score={data.get('score', '')}")
    lines.extend(["", "## Blocking Items"])
    if manifest["status"]["blocking_items"]:
        lines.extend(f"- {item}" for item in manifest["status"]["blocking_items"])
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def main() -> None:
    root_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--asset-dir", type=Path)
    parser.add_argument("--long-run-evidence", type=Path)
    parser.add_argument("--remote-preflight", type=Path)
    parser.add_argument("--json-out", type=Path, default=Path(__file__).resolve().parents[1] / "docs/submission_manifest.json")
    parser.add_argument("--md-out", type=Path, default=Path(__file__).resolve().parents[1] / "docs/submission_manifest.md")
    args = parser.parse_args()
    default_asset_dir = root_default.parents[0] / "harness-assets" / "battery_soh_rul"
    args.asset_dir = args.asset_dir or resolve_asset_dir(default_asset_dir, [REMOTE_ASSET_DIR])
    manifest = build_submission_manifest(args.root, args.asset_dir, args.long_run_evidence, args.remote_preflight)
    args.json_out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.md_out.write_text(_to_markdown(manifest), encoding="utf-8")
    print(json.dumps(manifest["status"], indent=2, ensure_ascii=False))
    raise SystemExit(0 if manifest["status"]["final_submission_ready"] else 2)


if __name__ == "__main__":
    main()
