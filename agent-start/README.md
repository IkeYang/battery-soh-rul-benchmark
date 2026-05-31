# Battery SOH/RUL and Anomalous-Cycle Modeling

You are building a battery analytics pipeline for an energy-storage validation lab. The dataset contains cycle-level telemetry from many lithium-ion cells operated under mixed charge protocols, temperatures, state-of-charge windows, and injected field-like faults.

Your job is to improve `train.py`, `predict.py`, and `src/` so the hidden evaluation set is predicted well. The training data is in `dev-data/train/`. A visible validation split is provided as `dev-data/dev_features.csv` and `dev-data/dev_labels.csv`; use it for local experiments only. The final judge will retrain from a clean state and then run your `predict.py` on hidden feature rows.

Output must be a CSV with exactly these columns:

`cell_id, cycle_index, predicted_soh, predicted_rul_cycles, predicted_anomaly_type, predicted_anomaly_severity`

`predicted_anomaly_type` must be one of:

`normal, capacity_drop, resistance_spike, thermal_event, sensor_drift, recovery_relaxation`

Higher score is better. The hidden scorer rewards low SOH error, early-life and knee-region RUL accuracy, EOL and knee-cycle calibration, anomalous-cycle detection, anomaly type classification, and severity ranking. Ordinary global regressors should make progress, but strong scores require robust feature engineering, per-chemistry/protocol behavior, fault-aware modeling, and careful validation.

Recommended workflow:

- Submit a working baseline early with `sebench-submit`, then iterate. Do not spend the first pass on a large rewrite without a scored checkpoint.
- Use `python local_validate.py --sample-train-cells 120 --sample-dev-cells 40` for quick comparisons.
- Run full `python local_validate.py` before final submissions. It reports visible-dev metrics, deterministic `stress_*` sensor/protocol perturbation metrics, harder `regime_stress_*` lifetime/EOL perturbations, and `challenge_*` hidden-like diagnostics with wider EOL-gap variance, capacity sensor calibration drift, and denser anomaly windows. These transforms are built only from visible labels; they are not hidden answers.
- Favor changes that improve `robust_raw_quality`, which is the minimum of visible, stress, regime-stress, and challenge raw quality. Keep `visible_to_stress_drop`, `visible_to_regime_stress_drop`, and `visible_to_challenge_drop` small. A visible-dev score jump that hurts stress or challenge validation is likely overfitting.
- Hidden cells may include lifetime-regime and capacity-calibration shifts, so avoid relying only on observed capacity ratio or the visible dev split. Combine SOH proxies with EOL/RUL calibration, anomaly-specific local-window signals, and conservative thresholds.
- Strong CPU-friendly routes usually include all of the following: cell-level EOL-gap modeling from aggregate degradation features, a capacity-calibrated SOH model blended with non-capacity curve features, local rolling-window anomaly features, a balanced anomaly classifier plus separate severity regressor, and small staged ensembles such as HistGradientBoosting, ExtraTrees, RandomForest, or calibrated linear fallbacks. Watch `challenge_metrics["anomaly_type_accuracy_on_anomaly"]`, `challenge_metrics["severity_rmse_on_anomaly"]`, and `challenge_metrics["eol_mae"]`; these often expose improvements that visible dev alone hides.
- Keep a known-good, fast version submitted before heavy experiments. Training has an evaluation timeout; large forests or many full-data ensembles can turn a promising solution into a zero-score submission.

Rules:

- Do not use external network data.
- Keep the CLI interfaces of `train.py` and `predict.py` working.
- Do not assume access to hidden labels or scorer internals.
- Use CPU-friendly methods; GPU dependencies are unnecessary.
