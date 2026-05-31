# Battery SOH/RUL Anomaly SE-Bench Task

This repository contains a SE-Bench Research task for lithium-ion battery health modeling. The agent receives a CPU-only Python project and visible development data. It must improve the training and prediction pipeline for hidden cell-cycle data.

The task combines three objectives:

- Predict cycle-level state of health (`SOH`).
- Predict remaining useful life in cycles (`RUL`), including early-life and knee-region behavior.
- Detect anomalous cycles and classify anomaly type/severity.

The benchmark data is a deterministic private derivative dataset.

Current local calibration:

- Starter grouped-statistics baseline: `3.520740`.
- Ordinary HistGradientBoosting reference: `9.780420`.
- Perfect hidden-label submission: `~100`.

This iteration targets a stepwise long-run curve: 30m remains at or below 15, 1h/2h stay at or below 30, and a full 8h agent run should land in the 40-50 range. The final target is not complete until harness assets, remote smoke/30m/1h/2h/8h validation runs, and submission docs are regenerated for this iteration.

Iteration 13 starts from Iter12 after true remote runs scored `13.97` at 30m, `28.96` at 1h, `15.00` at 2h, and `31.86` at 8h. The 8h run reached raw_quality `0.775292`, so Iter13 keeps the data and prompt fixed and retunes only the 8h band: raw_quality `0.773535` remains the two-hour acceptance ceiling at score `30`, and raw_quality `0.775292` maps into the 40-50 target band.

## GitHub and Release Layout

This GitHub repository intentionally keeps large benchmark data out of normal git history. The runnable SE-Bench assets are published as GitHub Release files:

- `battery_work.tar.gz`: work container payload visible to the agent.
- `battery_judge.tar.gz`: judge container payload with hidden evaluation data and scorer.
- `SHA256SUMS`: checksums for both assets.

Expected release URLs after upload:

```text
https://github.com/IKE1997/battery-soh-rul-benchmark/releases/download/v0.1.0/battery_work.tar.gz
https://github.com/IKE1997/battery-soh-rul-benchmark/releases/download/v0.1.0/battery_judge.tar.gz
https://github.com/IKE1997/battery-soh-rul-benchmark/releases/download/v0.1.0/SHA256SUMS
```

Set `BATTERY_WORK_ASSET_URL` and `BATTERY_JUDGE_ASSET_URL` to these URLs when building/running the harness outside the local asset server.
