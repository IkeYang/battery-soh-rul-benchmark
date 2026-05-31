# Battery Benchmark Pre-Submission Review

## Current State

- Deterministic generator implemented.
- Scorer implemented with structured JSON-compatible output.
- Harness output also emits `score_sum` case lines for SE-Bench 0.2.0 compatibility.
- Agent starter pipeline implemented and runnable.
- Unit tests cover hidden-label separation, deterministic generation, perfect scoring, and format rejection.
- Local starter and HGB reference baselines have been calibrated.
- Remote work/judge images, no-op harness smoke, HGB reference harness smoke, live SeedEdge preflight, and 30m/1h/2h/8h Codex validation have been verified on the VM.
- Asset packaging is deterministic: repeated packaging of the same tree produces identical tarball hashes.

## Difficulty Calibration

Current scores on the local hidden eval split:

- Starter grouped-statistics baseline: `3.520740`.
- Ordinary HGB multitask reference: `10.652701`.
- Perfect submission from hidden labels: at least `99.9`.

The current asset hashes are recorded outside the tarballs in `harness-assets/battery_soh_rul/SHA256SUMS`.

The current curve reserves 30+ points for stronger work beyond ordinary HGB: anomaly-specialized modeling, per-chemistry/protocol treatment, robust knee-cycle calibration, severity ranking, and ensembles.

## Completed Validation

Current final-submission checks:

- Local scorer and deterministic package audit pass for the current asset hashes.
- VM `python tools/long_run_preflight.py --require-live-auth` returns `ok: true`; the stored evidence records only safe credential metadata.
- Current asset long-run evidence is complete in `docs/long_run_evidence.json`.
- 30m validation: `battery_agent_validation_30m_newasset_20260528_130830`, score `14.028992`, runtime `1800.029s`.
- 1h validation: `battery_agent_validation_1h_longwrap_20260528_163804`, score `13.578930`, runtime `3600.029s`.
- 2h validation: `battery_agent_validation_2h_longwrap_20260528_174120`, score `17.770383`, runtime `7200.029s`.
- 8h validation: `battery_agent_validation_8h_longwrap_20260528_194716`, score `21.929065`, runtime `28800.030s`.

## Remote VM Validation Notes

As of 2026-05-29, the remote VM has the task JSON, asset HTTP server, judge server, Codex SeedEdge proxy patch, live SeedEdge preflight, and the full validation ladder working. Earlier remote no-op/HGB smoke evidence and aborted stale-asset runs used a package with `train_rows=427155` and `eval_rows=254230`; those runs are invalid for the current package. The current package has `train_rows=486003`, `dev_rows=109997`, and `eval_rows=289148`.
