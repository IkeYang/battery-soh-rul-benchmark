#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TASK_ID = "battery_soh_rul_anomaly"
STAGE_RULES = {
    "30m": {"min_runtime_seconds": 1200, "max_score": 15.0},
    "1h": {"min_runtime_seconds": 3000, "max_score": 30.0},
    "2h": {"min_runtime_seconds": 6600, "max_score": 30.0},
    "8h": {"min_runtime_seconds": 24000, "min_score": 40.0, "max_score": 50.0},
}
REQUIRED_STAGES = ["30m", "1h", "2h", "8h"]


def _load_final_result(log_root: Path, run_id: str) -> tuple[dict[str, Any] | None, Path]:
    path = log_root / run_id / TASK_ID / "final_result.json"
    if not path.exists():
        return None, path
    return json.loads(path.read_text(encoding="utf-8")), path


def _evaluate_stage(stage: str, run_id: str, result: dict[str, Any] | None, path: Path) -> dict[str, Any]:
    if result is None:
        return {"stage": stage, "run_id": run_id, "status": "missing", "path": str(path), "reasons": ["final_result.json not found"]}
    rules = STAGE_RULES[stage]
    score = float(result.get("best_score", 0.0))
    runtime = float(result.get("runtime_seconds", 0.0))
    reasons = []
    if runtime < float(rules["min_runtime_seconds"]):
        reasons.append("runtime below stage minimum")
    max_score = rules["max_score"]
    min_score = rules.get("min_score")
    if min_score is not None and score < float(min_score):
        reasons.append(f"score below {stage} target floor")
    if max_score is not None and score > float(max_score):
        reasons.append(f"score exceeds {stage} ceiling")
    status = "pass" if not reasons else "fail"
    return {
        "stage": stage,
        "run_id": run_id,
        "status": status,
        "path": str(path),
        "score": score,
        "runtime_seconds": runtime,
        "timed_out": bool(result.get("timed_out", False)),
        "best_round": result.get("best_round", ""),
        "total_rounds": int(result.get("total_rounds", 0)),
        "reasons": reasons,
    }


def collect_long_run_evidence(log_root: Path, stage_run_ids: dict[str, str]) -> dict[str, Any]:
    log_root = Path(log_root)
    runs = {}
    for stage in REQUIRED_STAGES:
        run_id = stage_run_ids.get(stage)
        if not run_id:
            runs[stage] = {"stage": stage, "run_id": "", "status": "missing", "reasons": ["run id not provided"]}
            continue
        result, path = _load_final_result(log_root, run_id)
        runs[stage] = _evaluate_stage(stage, run_id, result, path)
    missing = [stage for stage, item in runs.items() if item["status"] == "missing"]
    failed = [stage for stage, item in runs.items() if item["status"] == "fail"]
    completed = not missing and not failed
    return {
        "completed": completed,
        "runs": runs,
        "missing": missing,
        "failed": failed,
        "rules": STAGE_RULES,
    }


def _parse_stage_run(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected STAGE=RUN_ID, got {value!r}")
        stage, run_id = value.split("=", 1)
        if stage not in STAGE_RULES:
            raise ValueError(f"unknown stage {stage!r}")
        result[stage] = run_id
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, default=Path("/root/SE-bench-main/logs/runs"))
    parser.add_argument("--stage-run", action="append", default=[], help="Stage mapping like 30m=run_id. Repeat for 1h, 2h, 8h.")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "docs/long_run_evidence.json")
    args = parser.parse_args()
    evidence = collect_long_run_evidence(args.log_root, _parse_stage_run(args.stage_run))
    args.out.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"completed": evidence["completed"], "missing": evidence["missing"], "failed": evidence["failed"]}, indent=2, ensure_ascii=False))
    raise SystemExit(0 if evidence["completed"] else 2)


if __name__ == "__main__":
    main()
