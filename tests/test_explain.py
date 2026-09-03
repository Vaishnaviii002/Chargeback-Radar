import json

import pandas as pd
import pytest

from src.explain import (
    explain_payment,
    load_model_bundle,
)


@pytest.fixture(scope="module")
def sample_payment() -> dict:
    features = pd.read_parquet(
        "data/features.parquet"
    )

    return (
        features.sort_values("created_at")
        .iloc[-1]
        .to_dict()
    )


def test_explanation_returns_requested_factors(
    sample_payment,
):
    result = explain_payment(
        sample_payment,
        top_n=6,
    )

    assert result["method"] == "TreeSHAP"
    assert len(result["top_factors"]) == 6


def test_raw_probability_is_valid(
    sample_payment,
):
    result = explain_payment(sample_payment)

    assert (
        0
        <= result["raw_model_probability"]
        <= 1
    )


def test_factors_are_sorted_by_importance(
    sample_payment,
):
    result = explain_payment(
        sample_payment,
        top_n=10,
    )

    importance = [
        factor["absolute_importance"]
        for factor in result["top_factors"]
    ]

    assert importance == sorted(
        importance,
        reverse=True,
    )


def test_factor_directions_match_shap_values(
    sample_payment,
):
    result = explain_payment(
        sample_payment,
        top_n=10,
    )

    for factor in result["top_factors"]:
        if factor["shap_value"] > 0:
            assert (
                factor["direction"]
                == "INCREASES_RISK"
            )
        else:
            assert (
                factor["direction"]
                == "DECREASES_RISK"
            )


def test_explanation_is_json_serializable(
    sample_payment,
):
    result = explain_payment(sample_payment)

    serialized = json.dumps(result)

    assert "TreeSHAP" in serialized
    assert "top_factors" in serialized


def test_future_outcomes_are_never_explained(
    sample_payment,
):
    result = explain_payment(
        sample_payment,
        top_n=20,
    )

    forbidden_features = {
        "chargeback_within_120d",
        "chargeback_family",
        "chargeback_reason_code",
        "dispute_created_at",
        "dispute_status",
    }

    explained_features = {
        factor["feature"]
        for factor in result["top_factors"]
    }

    assert explained_features.isdisjoint(
        forbidden_features
    )


def test_missing_model_feature_is_rejected(
    sample_payment,
):
    bundle = load_model_bundle()
    incomplete_payment = sample_payment.copy()

    missing_feature = bundle[
        "model_features"
    ][0]

    incomplete_payment.pop(missing_feature)

    with pytest.raises(
        ValueError,
        match="Missing model features",
    ):
        explain_payment(incomplete_payment)


@pytest.mark.parametrize(
    "invalid_top_n",
    [0, 21, -1],
)
def test_invalid_top_n_is_rejected(
    sample_payment,
    invalid_top_n,
):
    with pytest.raises(
        ValueError,
        match="top_n must be between 1 and 20",
    ):
        explain_payment(
            sample_payment,
            top_n=invalid_top_n,
        )