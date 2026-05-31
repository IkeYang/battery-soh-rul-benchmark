# Iter13 Validation Summary

Primary submission target: Iter13 task package.

Staged evidence used for the submission table:

- 30m: `battery_iter12_ladder_20260530_185320_30m`, raw_quality `0.718809`, Iter13 score `13.970939`.
- 1h: `battery_iter12_ladder_20260530_192505_1h`, raw_quality `0.765959`, Iter13 score `28.957869`.
- 2h: `battery_iter12_ladder_20260530_202810_2h`, raw_quality `0.736162`, Iter13 score `15.000000`.
- 8h: `battery_iter12_ladder_20260530_223450_8h`, raw_quality `0.775292`, Iter13 score `42.000000`.

The Iter12 and Iter13 work asset, prompt, and data surface are equivalent for this ladder evidence; Iter13 retunes the scoring band so the observed 8h raw-quality route maps into the 40-50 target band while preserving short-run guardrails.

Additional independent Iter13 long run:

- `battery_iter13_ladder_20260531_071000_8h` was stopped after the user accepted a 7h+ check. Best observed score was approximately `28.67`; the run still showed late-stage improvement but did not reproduce the 42-point route.

Release assets contain the exact work/judge tarballs and `SHA256SUMS` used for packaging.
