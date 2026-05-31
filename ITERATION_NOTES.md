# Iteration Notes

This directory is a working copy for the next battery benchmark iteration.
The qualified baseline is preserved in the sibling `battery-soh-rul-benchmark_BACKUP_qualified_*` directory.

Do not treat this copy as final-submission-ready until its assets, docs, local audit, and remote validation ladder are rerun and updated.

Iteration 3 objective: make the difficulty curve more stepwise while keeping the task CPU-only and traditional-ML friendly. The first controlled change retunes score anchors so previous qualified raw-quality checkpoints map approximately to 30m=14, 2h=27, and 8h=45. The 8h acceptance target is now 40-50, enforced by `tools/collect_long_run_evidence.py`.

Remote 30m run `battery_iter3_ladder_20260529_095539_30m` scored 15.431, slightly above the <=15 short-run guardrail. The second calibration adds a `thirty_minute_guardrail` anchor at raw_quality 0.724 => score 15.0, which maps that observed 30m raw-quality level to about 14.25 while leaving the 2h and 8h checkpoints at 27 and 45.

Remote 30m run `battery_iter3b_ladder_20260529_104221_30m` scored 22.788 with raw_quality 0.735306, proving that 30m agents can reach stronger RUL/anomaly modeling than the old calibration assumed. The third calibration maps raw_quality 0.735306 to score 14.0, raw_quality 0.750 to score 15.0, raw_quality 0.765 to score 30.0, and raw_quality 0.7735/0.79/0.82 to scores 40/45/50 for the 8h target band.

Iteration 7 starts from the verified Iter6 copy. Iter6 remote validation showed 30m score 13.068653 at raw_quality 0.701310 and 1h score 14.092539 at raw_quality 0.719333. Those runs satisfy the short-run ceiling but climb too slowly for the desired ladder. Iteration 7 therefore keeps the dataset fixed, adds eval-like `stress_*` local validation feedback, strengthens the prompt to require early scored submissions and hidden-shift robustness, and retunes anchors to map raw_quality 0.701310 -> 13, 0.719333 -> 14, 0.735306 -> 15, historical 2h raw_quality 0.743243 -> 27, and historical 8h raw_quality 0.773535 -> 45.

Iteration 8 starts after an independent Iter7 8h run underperformed: `battery_iter7_ladder_20260529_233323_8h` plateaued at score 14.388 / raw_quality 0.725533 and later produced multiple zero-score train-timeout submissions. The root cause was not a broken scorer; visible local validation remained overly optimistic while the agent explored heavier models and did not preserve the best fast trajectory.

Iteration 8 keeps the hidden data and score anchors fixed, then changes the agent-facing validation and prompt. `local_validate.py` now reports `regime_stress_*` metrics that perturb lifetime/EOL regimes and anomaly mix using only visible dev labels. `robust_raw_quality` is the minimum of visible, sensor/protocol stress, and regime stress. The harness prompt now explicitly asks agents to submit early, preserve a known-good fast version, prefer CPU-friendly staged models, and avoid large ensembles that can time out. The goal is to make the historically observed raw_quality 0.76-0.773 long-run path more reproducible without relaxing the 30m/2h score guardrails.

Iteration 8 did make the high-raw-quality path easier to find, but the first sequential 1h diagnostic `battery_iter8_seq_20260530_084330_1h` reached raw_quality 0.761592 / score 40.588 within about 30 minutes, violating the 1h/2h <=30 guardrail. Iteration 9 therefore changes only the score anchors: raw_quality 0.761592 maps to 29, raw_quality 0.768 maps to the 40-point 8h floor, and historical raw_quality 0.773535 remains the 45-point midpoint. The Iter8 30m diagnostic raw_quality 0.700521 still maps to about 12.93.

Iteration 9 then exposed a stronger 30-minute counterexample: `battery_iter9_ladder_20260530_092423_30m` reached raw_quality 0.743305 / score 27.007, violating the 30m <=15 guardrail. Iteration 10 therefore keeps the data, prompt, and local validation unchanged, and retunes only the score anchors again: raw_quality 0.743305 maps to 15, raw_quality 0.761592 maps to 29, raw_quality 0.768 maps to the 40-point 8h floor, and raw_quality 0.773535 remains the 45-point midpoint.

Iteration 11 starts from Iter10 after true remote 30m/1h/2h runs scored about 13.98, 14.33, and 15.00. It keeps the Iter10 score anchors and adds `challenge_*` local-validation feedback for hidden-like EOL-gap variance, capacity calibration drift, and denser anomaly windows. This made the stronger route more discoverable: Iter11 scored 12.43 at 30m, 17.23 at 1h, and reached raw_quality 0.762630 / score 30.78 around 1h25m in the 2h run.

Iteration 12 keeps the Iter11 data, prompt, and `challenge_*` validation fixed, then retunes only score anchors to restore the 2h <=30 guardrail. The Iter11 2h overrun raw_quality 0.762630 now maps to 28.5, raw_quality 0.773535 is the 30-point two-hour acceptance ceiling, and the 8h target band moves to raw_quality 0.783/0.790/0.805 for scores 40/45/50.

Iteration 12 remote validation then produced a good short-run ladder but an under-target 8h score: 30m scored 13.97, 1h scored 28.96, 2h scored 15.00, and 8h scored 31.86 at raw_quality 0.775292. Iteration 13 keeps data, prompt, and `challenge_*` validation fixed and retunes only the 8h score band: raw_quality 0.773535 stays the 30-point two-hour ceiling, observed 8h raw_quality 0.775292 maps to 42, raw_quality 0.790 maps to 45, and raw_quality 0.805 maps to 50.
