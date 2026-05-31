# Battery SOH/RUL Benchmark Design And Validation Plan

## Design

The task is a CPU-only ML research benchmark. The agent modifies a stable `train.py`/`predict.py` interface to model battery cycle telemetry. The hidden scorer evaluates three coupled capabilities:

- SOH regression, with extra pressure on knee-region accuracy.
- RUL/EOL calibration, with early-life RUL weighted separately.
- Anomalous-cycle detection, anomaly type classification, and severity ranking.

The generated dataset is a deterministic private derivative benchmark. Public sources are used as range and schema references; hidden labels are produced by the local generator through seeded degradation curves and fault injections. This avoids a trivial public-answer lookup path while keeping the data close to real battery-health workflows.

## Difficulty Shape

Current local hidden-eval calibration:

| Anchor | Score |
|---|---:|
| Starter grouped-statistics baseline | 3.520740 |
| Ordinary HGB multitask reference | 9.780420 |
| Iter6 30m raw-quality checkpoint | 13.0 |
| Iter6 1h raw-quality checkpoint | 14.0 |
| Short-run guardrail | 15.0 |
| Iter9 30m guardrail checkpoint | 15.0 |
| Iter8 1h guardrail checkpoint | 28.0 |
| Iter11 2h overrun guardrail | 28.5 |
| Two-hour acceptance ceiling | 30.0 |
| 8h target floor | 40.0 |
| 8h target midpoint | 45.0 |
| 8h target ceiling | 50.0 |
| Expert target | 100.0 |

The ordinary HGB reference intentionally stays below 10. Iteration 7 retuned the score anchors around observed Iter6 30m/1h runs and historical 2h/8h raw-quality checkpoints so progress is visibly stepwise. Iteration 8 improved the development signal after an independent Iter7 8h run plateaued at raw_quality 0.725533 and produced later train-timeout submissions. Iteration 9 lowered the raw_quality 0.761592 checkpoint after it appeared too early in a 1h diagnostic. Iteration 10 further lowers the raw_quality 0.743305 checkpoint to 15 after it appeared in a completed 30m diagnostic. Iteration 11 added `challenge_*` validation, which improved discoverability but let a 2h run reach raw_quality 0.762630 / score 30.78 around 1h25m. Iteration 12 restored the 2h guardrail but its 8h run scored only 31.86 at raw_quality 0.775292. Iteration 13 keeps raw_quality 0.773535 as the two-hour acceptance ceiling and maps observed 8h raw_quality 0.775292 into the 40-50 target band. To exceed the short-run baselines, an agent still needs anomaly-specific modeling, feature engineering, per-chemistry/protocol behavior, hidden-shift robustness, threshold tuning, and CPU-friendly ensemble or staged pipelines.

`agent-start/local_validate.py` now reports four raw-quality views:

- `raw_quality`: visible dev split.
- `stress_raw_quality`: deterministic sensor/protocol perturbation with visible labels fixed.
- `regime_stress_raw_quality`: deterministic lifetime/EOL regime and anomaly-mix perturbation using visible labels only.
- `challenge_raw_quality`: hidden-like EOL-gap, capacity-calibration, and denser anomaly-window perturbation using visible labels only.

`robust_raw_quality` is the minimum of those values. This makes local feedback less likely to reward visible-dev memorization and gives the agent concrete proxies for hidden lifetime-regime, capacity-calibration, and anomaly-density shifts without exposing hidden labels.

## Validation Ladder

1. Local unit tests: `python -m pytest tests -q`.
2. Local package scoring: `bash scorer/score.sh agent-start`.
3. Asset leak scan: verify work tar excludes `eval_labels.csv`, `baseline_metrics.json`, `split_manifest.json`, and hidden baseline submissions.
4. Harness smoke run after copying `harness/battery_soh_rul_anomaly.json` to `tasks/`. This is verified on the remote VM with parser `score_sum`.
5. Credential preflight on the VM: `python tools/long_run_preflight.py --require-live-auth`. This must return `ok: true` before any real agent run counts as validation evidence.
6. 30-minute agent run: score should show progress but remain <=15.
7. 1-hour and 2-hour agent runs: score should improve but remain <=30.
8. If the previous checks hold, run a full 8-hour validation and inspect all `report.json` files, not only `final_result.json`.

Previous validation evidence came from the qualified baseline asset. It is kept below as calibration history only; this iteration must rerun packaging, local audit, remote preflight, and the full validation ladder before it is final-submission-ready.

| Stage | Run ID | Runtime (s) | Score | Status |
|---|---|---:|---:|---|
| 30m | `battery_agent_validation_30m_newasset_20260528_130830` | 1800.029 | 14.028992 | pass |
| 1h | `battery_agent_validation_1h_longwrap_20260528_163804` | 3600.029 | 13.578930 | pass |
| 2h | `battery_agent_validation_2h_longwrap_20260528_174120` | 7200.029 | 17.770383 | stale-calibration |
| 8h | `battery_agent_validation_8h_longwrap_20260528_194716` | 28800.030 | 21.929065 | stale-calibration |
| 30m | `battery_iter6_ladder_20260529_212049_30m` | 1800.038 | 13.068653 | iter7-anchor-input |
| 1h | `battery_iter6_ladder_20260529_212049_1h` | 3600.037 | 14.092539 | iter7-anchor-input |

## Remote VM Notes

Use China-friendly mirrors through environment variables rather than hardcoding them in the task JSON:

```bash
export SEBENCH_PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export SEBENCH_APT_MIRROR_URL=http://mirrors.aliyun.com/debian
```

If the asset host or GitHub is slow, configure proxy variables externally:

```bash
export SEBENCH_HTTP_PROXY=http://127.0.0.1:7890
export SEBENCH_HTTPS_PROXY=http://127.0.0.1:7890
```

Current remote state on 2026-05-29: asset hosting, Docker images, judge server, live SeedEdge preflight, and the full validation ladder are working. Earlier remote no-op/HGB smoke and the aborted `battery_agent_validation_1h_fixed_20260528_115315` run used a stale asset package with `train_rows=427155` and `eval_rows=254230`; those runs are kept only as invalidation evidence and must not be used for the current difficulty. The current package has `train_rows=486003` and `eval_rows=289148`.

## Long-Run Command Template

After installing a valid project-provided `SEBENCH_AGENT_API_KEY`, copy the current task tree to the VM and run:

```bash
cd /root/SE-bench-main
python /root/battery-soh-rul-benchmark/tools/long_run_preflight.py --require-live-auth --run-id-prefix battery_agent_validation_$(date +%Y%m%d_%H%M%S)
```

The preflight prints the exact 30m, 1h, 2h, and 8h `uv run sebench run` commands. The template matches the current VM agent registry and uses:

- `--agent codex-seededge`
- `--model gpt-5.5-0424`
- `--eval-interval 300`
- `SEBENCH_AGENT_API_BASE_URL=https://api.seededge.top/v1`
- China-friendly Node/PyPI mirrors injected through environment variables

Do not paste the real API key into this repository, the task JSON, or any documentation.

After the staged runs finish, generate the machine-readable evidence file from SE-Bench logs:

```bash
cd /root/battery-soh-rul-benchmark
python tools/collect_long_run_evidence.py \
  --stage-run 30m=<30m_run_id> \
  --stage-run 1h=<1h_run_id> \
  --stage-run 2h=<2h_run_id> \
  --stage-run 8h=<8h_run_id>
python tools/submission_manifest.py
```

The collector enforces the current acceptance gates: 30m score must be <=15, 1h/2h scores must be <=30, 8h score must be in the 40-50 target band, and each run must produce `final_result.json` and meet the stage runtime floor. Harness budget exhaustion is recorded as `timed_out: true` because the default Stop Hook keeps the agent working until the configured timeout; this is valid evidence when the score/runtime gates pass.
