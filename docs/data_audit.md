# Battery Benchmark Data Audit

## Source And Derivation

The benchmark uses deterministic synthetic augmentation calibrated from public lithium-ion aging ranges and anomaly examples:

- NASA PCoE style cycle aging fields: capacity, resistance, temperature, charge duration.
- CALCE anomaly datasets as qualitative references for abnormal lots and fault-like behavior.
- MIT/Severson style cycle-life prediction framing: early-cycle data, cycle life, summary and cycle-level features.
- TRI metadata as a reference for modern protocol metadata, not as a direct copied evaluation source.

The generated evaluation labels are private. They are produced by the local generator using seeded degradation curves, chemistry/protocol shifts, knee-point variation, sensor drift, resistance spikes, thermal events, recovery events, and capacity-drop injections.

## Split Policy

The split is by cell, not by row. `train_*`, `dev_*`, and `eval_*` cell IDs are disjoint. The agent-visible package includes:

- `dev-data/train/cycle_features.csv`
- `dev-data/train/cycle_labels.csv`
- `dev-data/dev_features.csv`
- `dev-data/dev_labels.csv`
- `dev-data/sample_submission.csv`

The hidden judge package includes:

- `scorer/eval-data/eval_features.csv`
- `scorer/eval-data/eval_labels.csv`
- `scorer/eval-data/baseline_metrics.json`

`eval_labels.csv`, split-generation metadata, and baseline hidden submissions must not be included in the work package.

## Current Scale

The current generated local dataset has:

- 700 training cells, 486003 training rows.
- 160 development cells, 109997 development rows.
- 420 hidden evaluation cells, 289148 evaluation rows.

This is intentionally large enough to require real modeling but small enough for repeated CPU-only feedback cycles.

## Leakage Review

- Eval cell IDs use the `eval_` prefix and are separate from visible cells.
- Agent-visible `sample_submission.csv` is keyed to the visible dev split only.
- Hidden labels are absent from `agent-start/`.
- Public raw data files are not needed at runtime and should not be packaged into the agent work directory.
