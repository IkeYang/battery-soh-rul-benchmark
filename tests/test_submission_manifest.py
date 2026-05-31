from __future__ import annotations

import json
from pathlib import Path

from tools.submission_manifest import build_submission_manifest, resolve_asset_dir


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "task"
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    _write(
        root / "task.yaml",
        "\n".join(
            [
                "name: battery-soh-rul-anomaly",
                "title: 面向储能电池寿命评估的 SOH/RUL 预测与异常循环识别",
                "category: 工程/工科",
                "agent_type: LLM",
                "network_required: false",
                "gpu_required: false",
                "status:",
                "  closed_loop_verified: false",
                "  final_submission_ready: false",
            ]
        ),
    )
    _write(root / "harness/battery_soh_rul_anomaly.json", json.dumps({"task_id": "battery_soh_rul_anomaly"}))
    for doc in [
        "data_audit.md",
        "workload_and_authenticity.md",
        "pollution_control.md",
        "pre_submission_review.md",
        "environment_setup_cn.md",
        "design_and_validation_plan.md",
    ]:
        _write(root / "docs" / doc, f"# {doc}\n")
    _write(root / "baseline/starter-score-log.json", '>>>> Start Structured Result\n{"score": 5.7}\n>>>>> End Structured Result\n')
    _write(root / "baseline/reference-hgb-score-log.json", '{"score": 13.9}')
    _write(asset_dir / "SHA256SUMS", "abc  battery_work.tar.gz\ndef  battery_judge.tar.gz\n")
    (asset_dir / "battery_work.tar.gz").write_bytes(b"work")
    (asset_dir / "battery_judge.tar.gz").write_bytes(b"judge")
    return root, asset_dir


def test_submission_manifest_marks_not_ready_without_long_runs(tmp_path: Path) -> None:
    root, asset_dir = _fixture(tmp_path)

    manifest = build_submission_manifest(root, asset_dir)

    assert manifest["status"]["final_submission_ready"] is False
    assert manifest["status"]["closed_loop_verified"] is False
    assert manifest["validation"]["long_runs"]["completed"] is False
    assert "30m/1h/2h/8h agent validation is not complete" in manifest["status"]["blocking_items"]


def test_submission_manifest_collects_assets_baselines_and_required_docs(tmp_path: Path) -> None:
    root, asset_dir = _fixture(tmp_path)

    manifest = build_submission_manifest(root, asset_dir)

    assert manifest["task"]["task_id"] == "battery_soh_rul_anomaly"
    assert manifest["assets"]["battery_work.tar.gz"]["sha256"] == "abc"
    assert manifest["assets"]["battery_judge.tar.gz"]["size_bytes"] == 5
    assert manifest["baselines"]["starter"]["score"] == 5.7
    assert manifest["baselines"]["reference_hgb"]["score"] == 13.9
    assert all(item["exists"] for item in manifest["required_docs"])


def test_resolve_asset_dir_prefers_existing_candidates(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    existing = tmp_path / "assets"
    existing.mkdir()
    (existing / "SHA256SUMS").write_text("", encoding="utf-8")

    assert resolve_asset_dir(missing, [existing]) == existing


def test_submission_manifest_includes_remote_preflight_blocker(tmp_path: Path) -> None:
    root, asset_dir = _fixture(tmp_path)
    preflight_path = root / "docs/remote_preflight.json"
    preflight_path.write_text(
        json.dumps(
            {
                "ok": False,
                "checks": {
                    "sebench_agent_credentials": {
                        "ok": False,
                        "key": {"set": True, "length": 21, "http_header_safe": False},
                    }
                },
                "blocking_reason": "No valid key",
            }
        ),
        encoding="utf-8",
    )

    manifest = build_submission_manifest(root, asset_dir, remote_preflight_path=preflight_path)

    assert manifest["validation"]["remote_preflight"]["ok"] is False
    assert "remote live auth preflight is not passing" in manifest["status"]["blocking_items"]


def test_submission_manifest_can_mark_ready_only_with_required_long_run_evidence(tmp_path: Path) -> None:
    root, asset_dir = _fixture(tmp_path)
    evidence = {
        "runs": {
            "30m": {"run_id": "r30", "score": 12.0, "status": "pass"},
            "1h": {"run_id": "r1h", "score": 18.0, "status": "pass"},
            "2h": {"run_id": "r2h", "score": 24.0, "status": "pass"},
            "8h": {"run_id": "r8h", "score": 31.0, "status": "pass"},
        }
    }
    evidence_path = root / "docs/long_run_evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    preflight_path = root / "docs/remote_preflight.json"
    preflight_path.write_text(json.dumps({"ok": True, "checks": {}}), encoding="utf-8")

    manifest = build_submission_manifest(
        root,
        asset_dir,
        long_run_evidence_path=evidence_path,
        remote_preflight_path=preflight_path,
    )

    assert manifest["validation"]["long_runs"]["completed"] is True
    assert manifest["status"]["closed_loop_verified"] is True
    assert manifest["status"]["final_submission_ready"] is True
