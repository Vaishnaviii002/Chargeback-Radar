from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from src.ablation import (
    ABLATION_PREDICTIONS_PATH,
    ABLATION_REPORT_PATH,
    DELTA_METRICS,
    REPORT_VERSION,
    SEED,
    SIGNAL_FEATURES,
    TARGET,
)
from src.evaluate import (
    calculate_ece,
    calculate_precision_at_fraction,
)


def _report() -> dict:
    assert ABLATION_REPORT_PATH.exists(), (
        "Run python -m src.ablation first"
    )

    with ABLATION_REPORT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def _predictions() -> pd.DataFrame:
    assert ABLATION_PREDICTIONS_PATH.exists(), (
        "Run python -m src.ablation first"
    )

    return pd.read_parquet(
        ABLATION_PREDICTIONS_PATH
    )


def test_report_identity_and_controls() -> None:
    report = _report()

    assert (
        report["report_version"]
        == REPORT_VERSION
    )
    assert report["target"] == TARGET
    assert report["seed"] == SEED
    assert report["synthetic_data"] is True

    controls = report["controls"]

    assert controls[
        "same_payment_rows"
    ] is True

    assert controls[
        "same_customer_disjoint_splits"
    ] is True

    assert controls["same_seed"] is True

    assert controls[
        "same_lightgbm_parameters"
    ] is True

    assert controls[
        "same_early_stopping_procedure"
    ] is True

    assert controls[
        "same_calibration_selection"
    ] is True

    assert controls[
        "same_evaluation_procedure"
    ] is True

    assert controls[
        "support_events_after_scoring_excluded"
    ] is True


def test_dataset_counts_match_authoritative_split() -> None:
    dataset = _report()[
        "dataset"
    ]

    assert dataset[
        "training_rows"
    ] == 59_048

    assert dataset[
        "calibration_rows"
    ] == 8_779

    assert dataset[
        "test_rows"
    ] == 12_173

    assert dataset[
        "training_positives"
    ] == 420

    assert dataset[
        "calibration_positives"
    ] == 71

    assert dataset[
        "test_positives"
    ] == 87


def test_feature_sets_differ_only_by_three_signals() -> None:
    report = _report()

    baseline = report[
        "baseline"
    ]

    enhanced = report[
        "enhanced"
    ]

    assert SIGNAL_FEATURES == [
        "intent_to_cancel",
        "non_receipt_complaint",
        "dissatisfaction",
    ]

    assert (
        enhanced["features"]
        == baseline["features"]
        + SIGNAL_FEATURES
    )

    assert (
        enhanced["feature_count"]
        == baseline["feature_count"]
        + 3
    )

    assert not (
        set(SIGNAL_FEATURES)
        & set(
            baseline["features"]
        )
    )


def test_baseline_reproduces_canonical_metrics() -> None:
    comparison = _report()[
        "canonical_baseline_comparison"
    ]

    assert comparison is not None

    assert (
        comparison[
            "canonical_selected_method"
        ]
        == comparison[
            "ablation_selected_method"
        ]
    )

    assert (
        comparison[
            "maximum_absolute_difference"
        ]
        == pytest.approx(
            0.0,
            abs=1e-12,
        )
    )

    for difference in comparison[
        "metric_differences"
    ].values():
        assert difference == pytest.approx(
            0.0,
            abs=1e-12,
        )


def test_predictions_align_with_held_out_ids() -> None:
    predictions = _predictions()

    canonical_test = pd.read_parquet(
        "reports/test_scored.parquet",
        columns=[
            "payment_id",
            TARGET,
        ],
    )

    assert len(predictions) == 12_173
    assert predictions[
        "payment_id"
    ].is_unique

    assert (
        predictions["payment_id"]
        .astype(str)
        .tolist()
        == canonical_test[
            "payment_id"
        ]
        .astype(str)
        .tolist()
    )

    assert (
        predictions[TARGET]
        .astype(int)
        .tolist()
        == canonical_test[TARGET]
        .astype(int)
        .tolist()
    )


def test_prediction_artifact_contains_no_identity() -> None:
    predictions = _predictions()

    forbidden = {
        "customer_id",
        "customer_name",
        "customer_email",
        "customer_phone",
        "archetype",
        "message_text",
        "support_event_id",
        "dispute_id",
        "dispute_status",
        "reason_code",
        "chargeback_family",
        "true_fraud",
    }

    assert not (
        set(predictions.columns)
        & forbidden
    )

    assert set(
        predictions.columns
    ) == {
        "payment_id",
        TARGET,
        "baseline_probability",
        "enhanced_probability",
    }


@pytest.mark.parametrize(
    ("variant", "probability_column"),
    [
        (
            "baseline",
            "baseline_probability",
        ),
        (
            "enhanced",
            "enhanced_probability",
        ),
    ],
)
def test_reported_metrics_recalculate_exactly(
    variant: str,
    probability_column: str,
) -> None:
    report = _report()
    predictions = _predictions()

    labels = predictions[
        TARGET
    ].to_numpy()

    probability = predictions[
        probability_column
    ].to_numpy()

    assert np.isfinite(
        probability
    ).all()

    assert (
        (
            probability >= 0
        )
        & (
            probability <= 1
        )
    ).all()

    metrics = report[
        variant
    ]["metrics"]

    threshold = metrics[
        "operating_threshold"
    ]

    predicted = (
        probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predicted,
        labels=[0, 1],
    ).ravel()

    precision_values, recall_values, _ = (
        precision_recall_curve(
            labels,
            probability,
        )
    )

    recalculated = {
        "average_precision": (
            average_precision_score(
                labels,
                probability,
            )
        ),
        "pr_auc": auc(
            recall_values[::-1],
            precision_values[::-1],
        ),
        "precision": precision_score(
            labels,
            predicted,
            zero_division=0,
        ),
        "recall": recall_score(
            labels,
            predicted,
            zero_division=0,
        ),
        "f1": f1_score(
            labels,
            predicted,
            zero_division=0,
        ),
        "brier_score": (
            brier_score_loss(
                labels,
                probability,
            )
        ),
        "expected_calibration_error": (
            calculate_ece(
                labels,
                probability,
            )
        ),
        "precision_at_1_percent": (
            calculate_precision_at_fraction(
                labels,
                probability,
                0.01,
            )
        ),
        "precision_at_5_percent": (
            calculate_precision_at_fraction(
                labels,
                probability,
                0.05,
            )
        ),
        "flagged_count": int(
            predicted.sum()
        ),
        "flagged_rate": float(
            predicted.mean()
        ),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
    }

    for metric, value in recalculated.items():
        assert metrics[
            metric
        ] == pytest.approx(
            value,
            abs=1e-12,
        )


def test_deltas_are_reported_honestly() -> None:
    report = _report()

    baseline = report[
        "baseline"
    ]["metrics"]

    enhanced = report[
        "enhanced"
    ]["metrics"]

    deltas = report[
        "delta_enhanced_minus_baseline"
    ]

    assert set(deltas) == set(
        DELTA_METRICS
    )

    for metric in DELTA_METRICS:
        expected = (
            enhanced[metric]
            - baseline[metric]
        )

        assert deltas[
            metric
        ] == pytest.approx(
            expected,
            abs=1e-12,
        )


def test_report_preserves_mixed_result() -> None:
    deltas = _report()[
        "delta_enhanced_minus_baseline"
    ]

    assert deltas[
        "average_precision"
    ] > 0

    assert deltas[
        "recall"
    ] > 0

    assert deltas[
        "brier_score"
    ] < 0

    # This negative result is intentionally preserved.
    assert deltas[
        "precision"
    ] < 0