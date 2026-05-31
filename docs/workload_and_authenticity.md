# Workload And Authenticity

## Authenticity

Battery SOH/RUL modeling is a realistic energy-storage validation task. Labs and fleet operators need to estimate state of health, remaining useful life, knee-point onset, and abnormal cycle behavior from cycle telemetry. The task mirrors practical battery analytics work: mixed protocols, chemistry-specific behavior, noisy measurements, sensor drift, and rare but high-impact abnormal cycles.

## Human Workload Estimate

A domain engineer would need more than 20 effective hours:

- 2-3h inspect schemas, leakage risks, and chemistry/protocol distributions.
- 3-5h build baseline SOH and RUL models with grouped validation.
- 3-5h identify and model knee-region behavior and EOL calibration.
- 4-6h design anomaly detection and type classification with class imbalance.
- 3-5h tune model families, features, thresholds, and severity ranking.
- 2-4h package reproducible training/prediction scripts and validate hidden-style splits.

The high ceiling comes from multi-objective tradeoffs: improving global SOH can hurt knee behavior; improving anomaly recall can increase false positives; RUL calibration differs across protocol and chemistry.

## Idle-Time Estimate

Current local scorer runtime is seconds to under a minute for baseline-scale submissions. Full harness eval is expected to stay under 5-10 minutes per submission on CPU, so environment idle time should remain well below 50% of agent runtime.
