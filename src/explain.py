from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import shap


MODEL_PATH = Path("artifacts/model_bundle.joblib")
FEATURES_PATH = Path("data/features.parquet")


FEATURE_LABELS = {
    "amount_paise": "Payment amount",
    "hour_of_day": "Payment hour",
    "day_of_week": "Day of week",
    "is_weekend": "Weekend payment",
    "is_digital_good": "Digital product",
    "descriptor_clarity_score": "Billing descriptor clarity",
    "phone_verified": "Phone verification",
    "email_verified": "Email verification",
    "account_age_days": "Account age",
    "has_prior_order": "Previous order history",
    "total_prior_orders": "Total prior orders",
    "prior_disputes_count": "Prior dispute history",
    "days_since_last_order": "Days since previous order",
    "txns_last_1h": "One-hour payment velocity",
    "txns_last_24h": "Twenty-four-hour payment velocity",
    "txns_last_7d": "Seven-day payment velocity",
    "amount_last_24h_paise": "Recent payment value",
    "device_is_new": "New device",
    "ip_country_matches_billing": "IP and billing-country match",
    "ip_is_proxy_or_vpn": "Proxy or VPN usage",
    "threeds_liability_shift": "3DS liability shift",
    "billing_shipping_distance_km": (
        "Billing-to-shipping distance"
    ),
    "card_network": "Card network",
    "product_category": "Product category",
    "cvv_result": "CVV verification result",
    "threeds_status": "3DS authentication status",
}


def json_safe(value: Any) -> Any:
    if value is None:
        return None

    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, float) and not np.isfinite(value):
        return None

    return value


@lru_cache(maxsize=1)
def load_model_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "artifacts/model_bundle.joblib is missing. "
            "Run: python -m src.train"
        )

    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_tree_explainer() -> shap.TreeExplainer:
    bundle = load_model_bundle()

    return shap.TreeExplainer(
        bundle["model"]
    )


def source_feature_name(
    transformed_name: str,
    categorical_features: list[str],
) -> str:
    for feature in sorted(
        categorical_features,
        key=len,
        reverse=True,
    ):
        prefix = f"{feature}_"

        if transformed_name.startswith(prefix):
            return feature

    return transformed_name


def positive_class_shap_values(
    shap_values: Any,
) -> np.ndarray:
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[-1])
    else:
        values = np.asarray(shap_values)

    # Some SHAP versions return:
    # rows × features × classes.
    if values.ndim == 3:
        if values.shape[-1] == 2:
            values = values[:, :, 1]
        elif values.shape[1] == 2:
            values = values[:, 1, :]

    if values.ndim == 1:
        values = values.reshape(1, -1)

    if values.ndim != 2:
        raise ValueError(
            "Unexpected SHAP output shape: "
            f"{values.shape}"
        )

    return values


def positive_class_base_value(
    expected_value: Any,
) -> float:
    values = np.asarray(
        expected_value,
        dtype=float,
    ).reshape(-1)

    if len(values) == 0:
        return 0.0

    return float(values[-1])


def factor_explanation(
    feature: str,
    value: Any,
    contribution: float,
) -> str:
    label = FEATURE_LABELS.get(
        feature,
        feature.replace("_", " ").title(),
    )

    direction = (
        "increased"
        if contribution > 0
        else "decreased"
    )

    return (
        f"{label} ({value}) {direction} this "
        "payment's model risk estimate."
    )


def explain_payment(
    payment: Mapping[str, Any],
    top_n: int = 6,
) -> dict[str, Any]:
    """
    Return local TreeSHAP explanations for one payment.

    SHAP explains the raw LightGBM model before probability
    calibration. It does not claim that a feature caused fraud.
    """

    if top_n < 1 or top_n > 20:
        raise ValueError(
            "top_n must be between 1 and 20."
        )

    bundle = load_model_bundle()
    preprocessor = bundle["preprocessor"]
    model = bundle["model"]

    model_features = bundle["model_features"]
    categorical_features = bundle[
        "categorical_features"
    ]
    transformed_names = bundle[
        "transformed_feature_names"
    ]

    missing_features = [
        feature
        for feature in model_features
        if feature not in payment
    ]

    if missing_features:
        raise ValueError(
            "Missing model features: "
            + ", ".join(missing_features)
        )

    model_input = pd.DataFrame(
        [
            {
                feature: payment[feature]
                for feature in model_features
            }
        ]
    )

    transformed_input = preprocessor.transform(
        model_input
    )

    raw_probability = float(
        model.predict_proba(
            transformed_input
        )[0, 1]
    )

    explainer = load_tree_explainer()

    raw_shap_values = explainer.shap_values(
        transformed_input
    )

    shap_matrix = positive_class_shap_values(
        raw_shap_values
    )

    row_values = shap_matrix[0]

    if len(row_values) != len(transformed_names):
        raise ValueError(
            "SHAP feature count does not match the "
            "saved transformed feature names."
        )

    grouped_contributions: dict[str, float] = {}

    for transformed_name, contribution in zip(
        transformed_names,
        row_values,
    ):
        original_feature = source_feature_name(
            transformed_name,
            categorical_features,
        )

        grouped_contributions[original_feature] = (
            grouped_contributions.get(
                original_feature,
                0.0,
            )
            + float(contribution)
        )

    sorted_factors = sorted(
        grouped_contributions.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    top_factors = []

    for rank, (feature, contribution) in enumerate(
        sorted_factors[:top_n],
        start=1,
    ):
        value = json_safe(payment.get(feature))

        top_factors.append(
            {
                "rank": rank,
                "feature": feature,
                "label": FEATURE_LABELS.get(
                    feature,
                    feature.replace(
                        "_",
                        " ",
                    ).title(),
                ),
                "value": value,
                "shap_value": round(
                    contribution,
                    6,
                ),
                "absolute_importance": round(
                    abs(contribution),
                    6,
                ),
                "direction": (
                    "INCREASES_RISK"
                    if contribution > 0
                    else "DECREASES_RISK"
                ),
                "explanation": factor_explanation(
                    feature,
                    value,
                    contribution,
                ),
            }
        )

    return {
        "method": "TreeSHAP",
        "model_version": bundle["model_version"],
        "raw_model_probability": round(
            raw_probability,
            8,
        ),
        "base_value_raw_margin": round(
            positive_class_base_value(
                explainer.expected_value
            ),
            6,
        ),
        "top_factors": top_factors,
        "explanation_space": (
            "SHAP values explain the LightGBM raw-margin "
            "output before probability calibration."
        ),
        "disclosure": (
            "These are model attribution signals, not proof "
            "that any individual feature caused a chargeback."
        ),
    }


def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "data/features.parquet is missing. "
            "Run the data pipeline first."
        )

    bundle = load_model_bundle()
    features = pd.read_parquet(FEATURES_PATH)

    sample = (
        features.sort_values(
            "created_at"
        )
        .iloc[-1]
        .to_dict()
    )

    explanation = explain_payment(
        sample,
        top_n=6,
    )

    print(
        json.dumps(
            explanation,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()