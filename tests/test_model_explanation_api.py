from __future__ import annotations

import json

import pandas as pd
from fastapi.testclient import TestClient

from src.api import app
from src.explanation_text import DISCLAIMER
from src.model_explanation_api import (
    EXPECTED_LABEL,
)


client = TestClient(app)


def _production_payment_id() -> str:
    report = pd.read_parquet(
        "reports/test_model_explanations.parquet",
        columns=["payment_id"],
    )

    return str(
        report.iloc[0][
            "payment_id"
        ]
    )


def test_deterministic_explanation_endpoint() -> None:
    payment_id = (
        _production_payment_id()
    )

    response = client.get(
        (
            "/api/model-explanations/"
            + payment_id
        )
    )

    assert response.status_code == 200

    payload = response.json()

    assert (
        payload["payment_id"]
        == payment_id
    )

    assert (
        0
        <= payload[
            "calibrated_probability"
        ]
        <= 1
    )

    assert (
        payload["risk_percentage"]
        == round(
            payload[
                "calibrated_probability"
            ]
            * 100,
            3,
        )
    )

    assert (
        payload["delivery_mode"]
        == "DETERMINISTIC"
    )

    assert (
        payload["provider"]
        == "deterministic_formatter"
    )

    assert payload[
        "fallback_used"
    ] is False

    assert payload[
        "fallback_reason"
    ] is None

    assert (
        payload["label"]
        == EXPECTED_LABEL
    )

    assert (
        payload["disclaimer"]
        == DISCLAIMER
    )

    assert (
        DISCLAIMER
        in payload["explanation"]
    )

    assert (
        payload[
            "model_explanation_is_evidence"
        ]
        is False
    )

    assert (
        payload[
            "evidence_copilot_separate"
        ]
        is True
    )

    assert (
        payload[
            "human_approval_required"
        ]
        is True
    )

    assert (
        payload["action_executed"]
        is False
    )


def test_factors_are_limited_and_direction_safe() -> None:
    response = client.get(
        (
            "/api/model-explanations/"
            + _production_payment_id()
        )
    )

    assert response.status_code == 200

    payload = response.json()

    positive = payload[
        "top_positive_factors"
    ]

    negative = payload[
        "top_negative_factors"
    ]

    assert len(positive) <= 3
    assert len(negative) <= 2

    for factor in positive:
        assert (
            factor["shap_value"]
            > 0
        )
        assert (
            factor["direction"]
            == "increases_risk"
        )

    for factor in negative:
        assert (
            factor["shap_value"]
            < 0
        )
        assert (
            factor["direction"]
            == "decreases_risk"
        )


def test_response_contains_no_identity_or_outcomes() -> None:
    response = client.get(
        (
            "/api/model-explanations/"
            + _production_payment_id()
        )
    )

    assert response.status_code == 200

    serialized = json.dumps(
        response.json()
    ).lower()

    forbidden = [
        "customer_id",
        "customer_name",
        "customer_email",
        "customer_phone",
        "dispute_id",
        "dispute_status",
        "reason_code",
        "chargeback_within_120d",
        "true_fraud",
        "chargeback_outcome",
    ]

    for field in forbidden:
        assert field not in serialized


def test_probability_matches_policy_report() -> None:
    payment_id = (
        _production_payment_id()
    )

    response = client.get(
        (
            "/api/model-explanations/"
            + payment_id
        )
    )

    policy = pd.read_parquet(
        "reports/test_policy.parquet"
    )

    expected = policy.loc[
        policy["payment_id"]
        == payment_id,
        "calibrated_probability",
    ].iloc[0]

    assert response.status_code == 200

    assert response.json()[
        "calibrated_probability"
    ] == float(expected)


def test_invalid_payment_id_is_rejected() -> None:
    response = client.get(
        "/api/model-explanations/not-valid"
    )

    assert response.status_code == 422

    assert response.json()[
        "detail"
    ] == "Invalid payment ID format."


def test_unknown_payment_id_returns_404() -> None:
    response = client.get(
        (
            "/api/model-explanations/"
            "pay_unknown_123"
        )
    )

    assert response.status_code == 404

    assert response.json()[
        "detail"
    ] == "Payment not found."


def test_response_schema_is_strict() -> None:
    response = client.get(
        (
            "/api/model-explanations/"
            + _production_payment_id()
        )
    )

    assert response.status_code == 200

    payload = response.json()

    assert set(payload) == {
        "payment_id",
        "calibrated_probability",
        "risk_percentage",
        "recommended_action",
        "model_version",
        "shap_explanation_version",
        "deterministic_text_version",
        "delivery_text_version",
        "delivery_mode",
        "provider",
        "provider_model",
        "response_id",
        "latency_ms",
        "fallback_used",
        "fallback_reason",
        "label",
        "explanation",
        "disclaimer",
        "top_positive_factors",
        "top_negative_factors",
        "model_explanation_is_evidence",
        "evidence_copilot_separate",
        "human_approval_required",
        "action_executed",
    }