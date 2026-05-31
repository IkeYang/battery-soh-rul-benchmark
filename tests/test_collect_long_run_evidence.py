from __future__ import annotations

import json
from pathlib import Path

from tools.collect_long_run_evidence import collect_long_run_evidence


def _write_final(log_root: Path, run_id: str, score: float, runtime: float, timed_out: bool = False) -> None:
    path = log_root / run_id / "battery_soh_rul_anomaly" / "final_result.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "task": "battery_soh_rul_anomaly",
                "run_id": run_id,
                "best_score": score,
                "runtime_seconds": runtime,
                "timed_out": timed_out,
                "best_round": "agent-1",
                "total_rounds": 3,
            }
        ),
        encoding="utf-8",
    )


def test_collect_long_run_evidence_passes_expected_ladder(tmp_path: Path) -> None:
    log_root = tmp_path / "runs"
    _write_final(log_root, "battery_30m", 12.0, 1801)
    _write_final(log_root, "battery_1h", 18.0, 3602)
    _write_final(log_root, "battery_2h", 24.0, 7201)
    _write_final(log_root, "battery_8h", 45.0, 28803)

    evidence = collect_long_run_evidence(
        log_root,
        {"30m": "battery_30m", "1h": "battery_1h", "2h": "battery_2h", "8h": "battery_8h"},
    )

    assert evidence["completed"] is True
    assert evidence["runs"]["30m"]["status"] == "pass"
    assert evidence["runs"]["2h"]["status"] == "pass"
    assert evidence["runs"]["8h"]["status"] == "pass"


def test_collect_long_run_evidence_fails_when_30m_score_too_high(tmp_path: Path) -> None:
    log_root = tmp_path / "runs"
    _write_final(log_root, "battery_30m", 16.0, 1801)

    evidence = collect_long_run_evidence(log_root, {"30m": "battery_30m"})

    assert evidence["completed"] is False
    assert evidence["runs"]["30m"]["status"] == "fail"
    assert "score exceeds 30m ceiling" in evidence["runs"]["30m"]["reasons"]


def test_collect_long_run_evidence_allows_harness_budget_timeout(tmp_path: Path) -> None:
    log_root = tmp_path / "runs"
    _write_final(log_root, "battery_30m", 14.0, 1800, timed_out=True)

    evidence = collect_long_run_evidence(log_root, {"30m": "battery_30m"})

    assert evidence["runs"]["30m"]["status"] == "pass"
    assert evidence["runs"]["30m"]["timed_out"] is True


def test_collect_long_run_evidence_fails_when_2h_score_too_high(tmp_path: Path) -> None:
    log_root = tmp_path / "runs"
    _write_final(log_root, "battery_2h", 31.0, 7201)

    evidence = collect_long_run_evidence(log_root, {"2h": "battery_2h"})

    assert evidence["completed"] is False
    assert evidence["runs"]["2h"]["status"] == "fail"
    assert "score exceeds 2h ceiling" in evidence["runs"]["2h"]["reasons"]


def test_collect_long_run_evidence_fails_when_8h_score_misses_target_window(tmp_path: Path) -> None:
    log_root = tmp_path / "runs"
    _write_final(log_root, "battery_8h", 34.0, 28803)

    evidence = collect_long_run_evidence(log_root, {"8h": "battery_8h"})

    assert evidence["completed"] is False
    assert evidence["runs"]["8h"]["status"] == "fail"
    assert "score below 8h target floor" in evidence["runs"]["8h"]["reasons"]


def test_collect_long_run_evidence_reports_missing_run(tmp_path: Path) -> None:
    evidence = collect_long_run_evidence(tmp_path / "runs", {"8h": "missing"})

    assert evidence["completed"] is False
    assert evidence["runs"]["8h"]["status"] == "missing"
