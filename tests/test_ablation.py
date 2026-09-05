from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ablation import (
    DELTA_METRICS,
    SIGNAL_FEATURES,
    TARGET,
    _evaluate_probabilities,
    _merge_support_signals,
    _metric_deltas,
    _model_parameters,
)
from src.train import SEED


def _features() -> pd.DataFrame:
    created_at = pd.to_datetime(
        [
            "2025-01-01T00:00:00Z",
            "2025-02-01T00:00:00Z",
            "2025-03-01T00:00:00Z",
        ],
        utc=True,
    )

    return pd.DataFrame(
        {
            "payment_id": [
                "pay_001",
                "pay_002",
                "pay_003",
            ],
            "customer_id": [
                "cust_001",
                "cust_002",
                "cust_003",
            ],
            "created_at": created_at,
            "month_index": [
                1,
                2,
                3,
            ],
            TARGET: [
                0,
                1,
                0,
            ],
        }
    )


def _signals() -> pd.DataFrame:
    scoring_at = (
        _features()["created_at"]
        + pd.Timedelta(
            days=7
        )
    )

    return pd.DataFrame(
        {
            "payment_id": [
                "pay_001",
                "pay_002",
                "pay_003",
            ],
            "scoring_at": scoring_at,
            "intent_to_cancel": pd.Series(
                [
                    False,
                    True,
                    False,
                ],
                dtype=bool,
            ),
            "non_receipt_complaint": pd.Series(
                [
                    False,
                    False,
                    True,
                ],
                dtype=bool,
            ),
            "dissatisfaction": pd.Series(
                [
                    False,
                    True,
                    False,
                ],
                dtype=bool,
            ),
            "provider": [
                "openai",
                "openai",
                "deterministic_fallback",
            ],
        }
    )


def test_signal_feature_names_are_exact() -> None:
    assert SIGNAL_FEATURES == [
        "intent_to_cancel",
        "non_receipt_complaint",
        "dissatisfaction",
    ]


def test_merge_preserves_payment_alignment() -> None:
    features = _features()
    signals = _signals()

    merged = _merge_support_signals(
        features,
        signals,
    )

    assert (
        merged["payment_id"].tolist()
        == features[
            "payment_id"
        ].tolist()
    )

    assert len(merged) == len(
        features
    )

    assert [
        bool(value)
        for value in merged[
            "intent_to_cancel"
        ]
    ] == [
        False,
        True,
        False,
    ]


def test_merge_imports_only_safe_columns() -> None:
    merged = _merge_support_signals(
        _features(),
        _signals(),
    )

    assert "provider" not in merged.columns
    assert "message_text" not in merged.columns
    assert "support_event_id" not in merged.columns
    assert "dispute_status" not in merged.columns
    assert "reason_code" not in merged.columns
    assert "true_fraud" not in merged.columns


def test_missing_signal_payment_is_rejected() -> None:
    signals = _signals().iloc[
        :-1
    ].copy()

    with pytest.raises(
        ValueError,
        match="payment alignment failed",
    ):
        _merge_support_signals(
            _features(),
            signals,
        )


def test_extra_signal_payment_is_rejected() -> None:
    signals = _signals()

    extra = signals.iloc[
        [0]
    ].copy()

    extra["payment_id"] = (
        "pay_extra"
    )

    signals = pd.concat(
        [
            signals,
            extra,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="payment alignment failed",
    ):
        _merge_support_signals(
            _features(),
            signals,
        )


def test_duplicate_signal_payment_is_rejected() -> None:
    signals = _signals()

    signals.loc[
        1,
        "payment_id",
    ] = "pay_001"

    with pytest.raises(
        ValueError,
        match="duplicate payment IDs",
    ):
        _merge_support_signals(
            _features(),
            signals,
        )


def test_outcome_field_in_signal_report_is_rejected() -> None:
    signals = _signals()

    signals[
        "chargeback_within_120d"
    ] = [
        0,
        1,
        0,
    ]

    with pytest.raises(
        ValueError,
        match="Forbidden fields",
    ):
        _merge_support_signals(
            _features(),
            signals,
        )


def test_non_boolean_signal_is_rejected() -> None:
    signals = _signals()

    signals[
        "intent_to_cancel"
    ] = [
        0,
        1,
        0,
    ]

    with pytest.raises(
        ValueError,
        match="boolean dtype",
    ):
        _merge_support_signals(
            _features(),
            signals,
        )


def test_wrong_scoring_timestamp_is_rejected() -> None:
    signals = _signals()

    signals.loc[
        0,
        "scoring_at",
    ] = (
        pd.Timestamp(
            "2025-01-20",
            tz="UTC",
        )
    )

    with pytest.raises(
        ValueError,
        match="payment-plus-7d",
    ):
        _merge_support_signals(
            _features(),
            signals,
        )


def test_model_parameters_match_training() -> None:
    parameters = (
        _model_parameters(
            scale_pos_weight=12.5
        )
    )

    assert parameters == {
        "objective": "binary",
        "n_estimators": 800,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_child_samples": 100,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 0.2,
        "scale_pos_weight": 12.5,
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": -1,
    }


def test_evaluation_contains_required_metrics() -> None:
    calibration_probability = (
        np.linspace(
            0.01,
            0.99,
            100,
        )
    )

    test_labels = np.array(
        [
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            1,
            0,
        ]
    )

    test_probability = np.array(
        [
            0.05,
            0.90,
            0.10,
            0.80,
            0.15,
            0.20,
            0.70,
            0.30,
            0.60,
            0.40,
        ]
    )

    metrics = _evaluate_probabilities(
        calibration_probability=(
            calibration_probability
        ),
        test_identifiers=np.array(
            [f"pay_{index}" for index in range(10)]
        ),
        test_labels=test_labels,
        test_probability=(
            test_probability
        ),
    )

    required = {
        "average_precision",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "brier_score",
        "expected_calibration_error",
        "precision_at_1_percent",
        "precision_at_5_percent",
        "operating_threshold",
        "flagged_count",
        "flagged_rate",
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
    }

    assert set(metrics) == required

    assert (
        metrics["true_positive"]
        + metrics["false_positive"]
        + metrics["false_negative"]
        + metrics["true_negative"]
        == len(test_labels)
    )


def test_delta_is_enhanced_minus_baseline() -> None:
    baseline_metrics = {
        metric: 1.0
        for metric in DELTA_METRICS
    }

    enhanced_metrics = {
        metric: 1.25
        for metric in DELTA_METRICS
    }

    result = _metric_deltas(
        baseline={
            "metrics": baseline_metrics
        },
        enhanced={
            "metrics": enhanced_metrics
        },
    )

    assert set(result) == set(
        DELTA_METRICS
    )

    for value in result.values():
        assert value == pytest.approx(
            0.25
        )
