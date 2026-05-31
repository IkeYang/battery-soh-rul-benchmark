import json
from pathlib import Path

import pandas as pd

from scorer.evaluate import evaluate_submission
from scorer.evaluate import _score_from_anchors
from tools.generate_battery_benchmark import BenchmarkConfig, generate_benchmark


def _make_dataset(tmp_path: Path) -> Path:
    generate_benchmark(
        BenchmarkConfig(
            output_dir=tmp_path,
            seed=456,
            dev_cells=4,
            train_cells=8,
            eval_cells=5,
            min_cycles=90,
            max_cycles=110,
        )
    )
    return tmp_path


def test_scorer_rewards_perfect_submission(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path)
    features = root / "scorer" / "eval-data" / "eval_features.csv"
    labels = root / "scorer" / "eval-data" / "eval_labels.csv"
    metrics = root / "scorer" / "eval-data" / "baseline_metrics.json"
    truth = pd.read_csv(labels)
    submission = truth.rename(
        columns={
            "soh": "predicted_soh",
            "rul_cycles": "predicted_rul_cycles",
            "anomaly_type": "predicted_anomaly_type",
            "anomaly_severity": "predicted_anomaly_severity",
        }
    )[
        [
            "cell_id",
            "cycle_index",
            "predicted_soh",
            "predicted_rul_cycles",
            "predicted_anomaly_type",
            "predicted_anomaly_severity",
        ]
    ]
    submission_path = root / "perfect.csv"
    submission.to_csv(submission_path, index=False)

    result = evaluate_submission(features, labels, submission_path, metrics)

    assert result["valid"] is True
    assert result["total_score"] >= 99.9
    assert result["metrics"]["soh_mae"] == 0.0
    assert result["metrics"]["anomaly_macro_f1"] == 1.0


def test_scorer_rejects_missing_columns(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path)
    features = root / "scorer" / "eval-data" / "eval_features.csv"
    labels = root / "scorer" / "eval-data" / "eval_labels.csv"
    metrics = root / "scorer" / "eval-data" / "baseline_metrics.json"
    bad_path = root / "bad.csv"
    pd.DataFrame({"cell_id": ["x"], "cycle_index": [1]}).to_csv(bad_path, index=False)

    result = evaluate_submission(features, labels, bad_path, metrics)

    assert result["valid"] is False
    assert result["total_score"] == 0.0
    assert "missing required columns" in result["summary"]


def test_scorer_outputs_structured_json_shape(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path)
    payload = json.loads((root / "scorer" / "eval-data" / "baseline_metrics.json").read_text())

    assert "score_anchors" in payload
    assert payload["score_anchors"][0]["name"] == "weak_baseline"
    assert payload["score_anchors"][-1]["name"] == "expert_target"


def test_score_anchor_curve_supports_stepwise_long_run_ladder(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path)
    metrics = json.loads((root / "scorer" / "eval-data" / "baseline_metrics.json").read_text())
    anchors = sorted(metrics["score_anchors"], key=lambda item: item["raw_quality"])
    scores = [item["score"] for item in anchors]
    assert scores == sorted(scores)

    # Raw-quality checkpoints are derived from the previous qualified run's
    # local baselines and remote 30m/2h/8h scores under the old anchor curve.
    starter_score = _score_from_anchors(0.3067466328195683, metrics)
    reference_hgb_score = _score_from_anchors(0.5805711178213633, metrics)
    observed_iter4_30m_score = _score_from_anchors(0.6957656715842326, metrics)
    observed_iter6_30m_score = _score_from_anchors(0.7013097249484304, metrics)
    observed_iter6_1h_score = _score_from_anchors(0.7193325395358704, metrics)
    observed_short_run_strong_score = _score_from_anchors(0.7353059943553691, metrics)
    previous_two_hour_progress_score = _score_from_anchors(0.7432429432923224, metrics)
    observed_iter8_30m_score = _score_from_anchors(0.700520672380867, metrics)
    observed_iter9_30m_score = _score_from_anchors(0.7433049347998876, metrics)
    observed_iter8_1h_score = _score_from_anchors(0.7615917042309768, metrics)
    observed_iter11_2h_overrun_score = _score_from_anchors(0.7626304050926715, metrics)
    two_hour_ceiling_score = _score_from_anchors(0.7735352840883063, metrics)
    eight_hour_floor_score = _score_from_anchors(0.7752919808318431, metrics)
    eight_hour_mid_score = _score_from_anchors(0.79000, metrics)
    eight_hour_ceiling_score = _score_from_anchors(0.80500, metrics)

    assert 2.0 <= starter_score <= 5.0
    assert 9.0 <= reference_hgb_score <= 12.0
    assert 12.0 <= observed_iter4_30m_score <= 15.0
    assert observed_iter6_30m_score == 13.0
    assert observed_iter6_1h_score == 14.0
    assert observed_short_run_strong_score == 15.0
    assert previous_two_hour_progress_score <= 15.0
    assert 12.0 <= observed_iter8_30m_score <= 14.0
    assert observed_iter9_30m_score <= 15.0
    assert observed_iter8_1h_score <= 30.0
    assert observed_iter11_2h_overrun_score <= 30.0
    assert two_hour_ceiling_score == 30.0
    assert 40.0 <= eight_hour_floor_score <= 45.0
    assert eight_hour_mid_score == 45.0
    assert eight_hour_ceiling_score == 50.0
