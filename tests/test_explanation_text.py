from __future__ import annotations

from src.explanation_text import (
    DISCLAIMER,
    build_deterministic_explanation,
)


def test_positive_factor_is_described_as_model_contribution() -> None:
    text = build_deterministic_explanation(
        positive_factors=[
            {
                "feature": "device_is_new",
                "value": 1,
                "shap_value": 0.42,
                "direction": "increases_risk",
            }
        ],
        negative_factors=[],
    )

    assert "new-device status" in text
    assert "increased the model's risk estimate" in text


def test_negative_factor_is_described_as_model_contribution() -> None:
    text = build_deterministic_explanation(
        positive_factors=[],
        negative_factors=[
            {
                "feature": "threeds_liability_shift",
                "value": 1,
                "shap_value": -0.31,
                "direction": "decreases_risk",
            }
        ],
    )

    assert "3DS liability-shift status" in text
    assert "decreased the model's risk estimate" in text


def test_disclaimer_is_always_present() -> None:
    text = build_deterministic_explanation(
        positive_factors=[],
        negative_factors=[],
    )

    assert DISCLAIMER in text
    assert "not dispute evidence" in text
    assert "do not establish causation" in text


def test_explanation_does_not_accuse_customer() -> None:
    text = build_deterministic_explanation(
        positive_factors=[
            {
                "feature": "device_is_new",
                "value": 1,
                "shap_value": 0.5,
                "direction": "increases_risk",
            }
        ],
        negative_factors=[],
    ).lower()

    forbidden_phrases = [
        "customer committed fraud",
        "customer is fraudulent",
        "this is fraud",
        "the customer intended",
        "the customer caused",
        "guilty",
    ]

    for phrase in forbidden_phrases:
        assert phrase not in text


def test_only_supplied_factors_appear() -> None:
    text = build_deterministic_explanation(
        positive_factors=[
            {
                "feature": "txns_last_24h",
                "value": 7,
                "shap_value": 0.25,
                "direction": "increases_risk",
            }
        ],
        negative_factors=[
            {
                "feature": "email_verified",
                "value": 1,
                "shap_value": -0.15,
                "direction": "decreases_risk",
            }
        ],
    )

    assert "recent 24-hour transaction velocity" in text
    assert "email verification status" in text

    assert "new-device status" not in text
    assert "proxy/VPN indicator" not in text


def test_values_are_preserved() -> None:
    text = build_deterministic_explanation(
        positive_factors=[
            {
                "feature": "billing_shipping_distance_km",
                "value": 123.45,
                "shap_value": 0.3,
                "direction": "increases_risk",
            },
            {
                "feature": "amount_paise",
                "value": 259900,
                "shap_value": 0.2,
                "direction": "increases_risk",
            },
        ],
        negative_factors=[],
    )

    assert "123.5 km" in text
    assert "₹2,599.00" in text


def test_factor_limits_are_respected() -> None:
    positive = [
        {
            "feature": "device_is_new",
            "value": 1,
            "shap_value": 0.5,
            "direction": "increases_risk",
        },
        {
            "feature": "txns_last_24h",
            "value": 8,
            "shap_value": 0.4,
            "direction": "increases_risk",
        },
        {
            "feature": "ip_is_proxy_or_vpn",
            "value": 1,
            "shap_value": 0.3,
            "direction": "increases_risk",
        },
    ]

    text = build_deterministic_explanation(
        positive_factors=positive,
        negative_factors=[],
        max_positive=2,
    )

    assert "new-device status" in text
    assert "recent 24-hour transaction velocity" in text
    assert "proxy/VPN indicator" not in text


def test_zero_factor_limits_are_allowed() -> None:
    text = build_deterministic_explanation(
        positive_factors=[
            {
                "feature": "device_is_new",
                "value": 1,
                "shap_value": 0.4,
                "direction": "increases_risk",
            }
        ],
        negative_factors=[],
        max_positive=0,
        max_negative=0,
    )

    assert (
        "No material model contribution factors were available."
        in text
    )

    assert DISCLAIMER in text


def test_negative_factor_limits_rejected() -> None:
    try:
        build_deterministic_explanation(
            positive_factors=[],
            negative_factors=[],
            max_positive=-1,
        )
    except ValueError as exc:
        assert "max_positive" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for negative max_positive"
        )


def test_unknown_feature_has_safe_fallback_label() -> None:
    text = build_deterministic_explanation(
        positive_factors=[
            {
                "feature": "some_new_feature",
                "value": 5,
                "shap_value": 0.2,
                "direction": "increases_risk",
            }
        ],
        negative_factors=[],
    )

    assert "some new feature" in text


def test_direction_is_preserved() -> None:
    text = build_deterministic_explanation(
        positive_factors=[
            {
                "feature": "device_is_new",
                "value": 1,
                "shap_value": 0.4,
                "direction": "increases_risk",
            }
        ],
        negative_factors=[
            {
                "feature": "email_verified",
                "value": 1,
                "shap_value": -0.2,
                "direction": "decreases_risk",
            }
        ],
    )

    assert "increased the model's risk estimate" in text
    assert "decreased the model's risk estimate" in text


def test_empty_factors_still_produce_safe_explanation() -> None:
    text = build_deterministic_explanation(
        positive_factors=[],
        negative_factors=[],
    )

    assert len(text) > 0

    assert (
        "No material model contribution factors were available."
        in text
    )

    assert DISCLAIMER in text