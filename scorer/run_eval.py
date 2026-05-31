#!/usr/bin/env python3
"""Run an agent submission and return structured JSON for SE-Bench."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from scorer.evaluate import evaluate_submission


def _invalid(summary: str, output: str = "") -> dict:
    return {
        "valid": False,
        "score": 0.0,
        "pass_rate": 0.0,
        "total_score": 0.0,
        "summary": summary,
        "metrics": {},
        "details": [{"name": "execution", "status": "FAILED", "message": output[-4000:]}],
    }


def _run(cmd: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)


def format_score_sum_lines(result: dict, total_cases: int = 10000) -> list[str]:
    score = float(result.get("score", 0.0) or 0.0)
    clamped = max(0.0, min(100.0, score))
    passed = int(round((clamped / 100.0) * total_cases))
    lines = []
    for idx in range(total_cases):
        if idx < passed:
            lines.append(f"CASE {idx:04d} OK score={clamped}")
        else:
            lines.append(f"CASE {idx:04d} WA score=0")
    lines.append(f"TOTAL_SCORE {score}")
    lines.append(f"CASES_TOTAL {total_cases}")
    return lines


def run_evaluation(
    agent_dir: Path,
    features_path: Path,
    labels_path: Path,
    metrics_path: Path,
    timeout_train: int = 900,
    timeout_predict: int = 600,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="battery_eval_") as tmp:
        tmp_path = Path(tmp)
        exec_agent = tmp_path / "agent-start"
        shutil.copytree(agent_dir, exec_agent, ignore=shutil.ignore_patterns("artifacts", "submission.csv", "__pycache__", "*.pyc"))
        submission_path = exec_agent / "submission.csv"
        python_bin = sys.executable
        try:
            train = _run([python_bin, "train.py", "--train-dir", "dev-data/train", "--model-dir", "artifacts"], exec_agent, timeout_train)
        except subprocess.TimeoutExpired as exc:
            return _invalid("train timed out", (exc.stdout or "") if isinstance(exc.stdout, str) else "")
        except Exception as exc:
            return _invalid(f"train crashed before completion: {type(exc).__name__}: {exc}")
        if train.returncode != 0:
            return _invalid(f"train failed with exit code {train.returncode}", train.stdout)
        try:
            predict = _run(
                [python_bin, "predict.py", "--input", str(features_path), "--output", str(submission_path), "--model-dir", "artifacts"],
                exec_agent,
                timeout_predict,
            )
        except subprocess.TimeoutExpired as exc:
            return _invalid("predict timed out", (exc.stdout or "") if isinstance(exc.stdout, str) else "")
        except Exception as exc:
            return _invalid(f"predict crashed before completion: {type(exc).__name__}: {exc}")
        if predict.returncode != 0:
            return _invalid(f"predict failed with exit code {predict.returncode}", predict.stdout)
        result = evaluate_submission(features_path, labels_path, submission_path, metrics_path)
        result.setdefault("details", [])
        result["details"].append({"name": "train_tail", "status": "INFO", "message": train.stdout[-2000:]})
        result["details"].append({"name": "predict_tail", "status": "INFO", "message": predict.stdout[-2000:]})
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--timeout-train", type=int, default=900)
    parser.add_argument("--timeout-predict", type=int, default=600)
    args = parser.parse_args()
    result = run_evaluation(args.agent_dir, args.features, args.labels, args.metrics, args.timeout_train, args.timeout_predict)
    print(">>>> Start Structured Result")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    print(">>>>> End Structured Result")
    for line in format_score_sum_lines(result):
        print(line)


if __name__ == "__main__":
    main()
