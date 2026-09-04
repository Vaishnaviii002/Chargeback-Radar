from __future__ import annotations

from typing import Any


FEATURE_LABELS = {
    "amount_paise": "payment amount",
    "hour_of_day": "transaction hour",
    "day_of_week": "day of week",
    "is_weekend": "weekend timing",
    "is_digital_good": "digital-goods purchase",
    "descriptor_clarity_score": "statement descriptor clarity",
    "phone_verified": "phone verification status",
    "email_verified": "email verification status",
    "account_age_days": "account age",
    "has_prior_order": "prior-order history",
    "total_prior_orders": "number of prior orders",
    "prior_disputes_count": "prior dispute history",
    "days_since_last_order": "time since previous order",
    "txns_last_1h": "recent one-hour transaction velocity",
    "txns_last_24h": "recent 24-hour transaction velocity",
    "txns_last_7d": "recent seven-day transaction velocity",
    "amount_last_24h_paise": "recent 24-hour transaction amount",
    "device_is_new": "new-device status",
    "ip_country_matches_billing": "IP-to-billing-country match",
    "ip_is_proxy_or_vpn": "proxy/VPN indicator",
    "threeds_liability_shift": "3DS liability-shift status",
    "billing_shipping_distance_km": "billing-to-shipping distance",
    "card_network": "card network",
    "product_category": "product category",
    "cvv_result": "CVV result",
    "threeds_status": "3DS status",
}


DISCLAIMER = (
    "This is a model explanation, not dispute evidence. "
    "The factors describe contributions to the model's risk estimate "
    "and do not establish causation, fraud, or customer intent."
)


def _format_value(
    feature: str,
    value: Any,
) -> str:
    if value is None:
        return "missing"

    if feature in {
        "amount_paise",
        "amount_last_24h_paise",
    }:
        try:
            rupees = float(value) / 100.0
            return f"₹{rupees:,.2f}"
        except (TypeError, ValueError):
            return str(value)

    if feature == "billing_shipping_distance_km":
        try:
            return f"{float(value):,.1f} km"
        except (TypeError, ValueError):
            return str(value)

    if feature == "descriptor_clarity_score":
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)

    if isinstance(value, bool):
        return "yes" if value else "no"

    return str(value)


def _factor_phrase(
    factor: dict[str, Any],
) -> str:
    feature = str(
        factor["feature"]
    )

    label = FEATURE_LABELS.get(
        feature,
        feature.replace("_", " "),
    )

    value = _format_value(
        feature,
        factor.get("value"),
    )

    direction = factor.get(
        "direction"
    )

    if direction == "increases_risk":
        return (
            f"{label} ({value}) increased the model's risk estimate"
        )

    if direction == "decreases_risk":
        return (
            f"{label} ({value}) decreased the model's risk estimate"
        )

    return (
        f"{label} ({value}) had a neutral contribution"
    )


def build_deterministic_explanation(
    positive_factors: list[dict[str, Any]],
    negative_factors: list[dict[str, Any]],
    max_positive: int = 3,
    max_negative: int = 2,
) -> str:
    if max_positive < 0:
        raise ValueError(
            "max_positive cannot be negative"
        )

    if max_negative < 0:
        raise ValueError(
            "max_negative cannot be negative"
        )

    selected_positive = positive_factors[
        :max_positive
    ]

    selected_negative = negative_factors[
        :max_negative
    ]

    positive_phrases = [
        _factor_phrase(factor)
        for factor in selected_positive
    ]

    negative_phrases = [
        _factor_phrase(factor)
        for factor in selected_negative
    ]

    sections: list[str] = []

    if positive_phrases:
        sections.append(
            "Risk-increasing factors: "
            + "; ".join(positive_phrases)
            + "."
        )

    if negative_phrases:
        sections.append(
            "Risk-reducing factors: "
            + "; ".join(negative_phrases)
            + "."
        )

    if not sections:
        sections.append(
            "No material model contribution factors were available."
        )

    sections.append(
        DISCLAIMER
    )

    return " ".join(
        sections
    )