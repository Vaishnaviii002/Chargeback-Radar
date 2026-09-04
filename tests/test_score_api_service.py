from datetime import datetime, timezone
import inspect

from fastapi.testclient import TestClient

import src.api as api_module
from src.api import app, score_payment
from src.risk_scoring import (
    RiskScoreResult,
    RiskScoringError,
)


UTC = timezone.utc
client = TestClient(app)


def _request_payload() -> dict:
    return {
        "created_at": datetime(
            2025,
            12,
            1,
            12,
            30,
            tzinfo=UTC,
        ).isoformat(),
        "amount_paise": 250000,
        "card_network": "Visa",
        "product_category": "electronics",
        "is_digital_good": False,
        "descriptor_clarity_score": 0.82,
        "phone_verified": True,
        "email_verified": True,
        "account_age_days": 420,
        "has_prior_order": 1,
        "total_prior_orders": 8,
        "prior_disputes_count": 0,
        "days_since_last_order": 20,
        "txns_last_1h": 0,
        "txns_last_24h": 1,
        "txns_last_7d": 2,
        "amount_last_24h_paise": 250000,
        "device_is_new": False,
        "ip_country_matches_billing": True,
        "ip_is_proxy_or_vpn": False,
        "cvv_result": "match",
        "threeds_status": "authenticated",
        "threeds_liability_shift": True,
        "billing_shipping_distance_km": 12,
        "is_duplicate_payment": False,
        "cancelled_subscription_billed": False,
    }


def _fake_result() -> RiskScoreResult:
    return RiskScoreResult(
        raw_probability=0.02,
        calibrated_probability=0.01,
        risk_percentage=1.0,
        model_recommended_action="MONITOR",
        recommended_action="MONITOR",
        decision_source="COST_OPTIMIZED_MODEL",
        rules={
            "triggered_rules": [],
            "hard_override_action": None,
        },
        risk_band="MEDIUM",
        expected_costs_rupees={
            "MONITOR": 100.0,
            "PREPARE_EVIDENCE": 120.0,
            "MANUAL_REVIEW": 180.0,
            "RECOMMEND_REFUND": 2500.0,
        },
        model_version="0.1.0",
        calibration_method="isotonic",
        requires_human_approval=False,
        action_executed=False,
    )


def test_score_endpoint_delegates_to_service(
    monkeypatch,
) -> None:
    captured = {}

    def fake_score(payload):
        captured["payload"] = payload
        return _fake_result()

    monkeypatch.setattr(
        api_module,
        "score_capture_time_payment",
        fake_score,
    )

    response = client.post(
        "/api/score",
        json=_request_payload(),
    )

    assert response.status_code == 200

    result = response.json()

    assert captured[
        "payload"
    ].amount_paise == 250000

    assert result[
        "calibrated_probability"
    ] == 0.01

    assert result[
        "recommended_action"
    ] == "MONITOR"

    assert result["action_executed"] is False


def test_endpoint_preserves_explanation_contract(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "score_capture_time_payment",
        lambda payload: _fake_result(),
    )

    response = client.post(
        "/api/score",
        json=_request_payload(),
    )

    explanation = response.json()[
        "explanation"
    ]

    assert explanation == {
        "available": False,
        "method": "TreeSHAP",
        "top_factors": [],
        "reason": (
            "Use the dedicated model-explanation endpoint "
            "for validated held-out transaction explanations."
        ),
    }


def test_scoring_failure_is_safely_normalized(
    monkeypatch,
) -> None:
    def fail(payload):
        raise RiskScoringError(
            "Internal detail must not be exposed."
        )

    monkeypatch.setattr(
        api_module,
        "score_capture_time_payment",
        fail,
    )

    response = client.post(
        "/api/score",
        json=_request_payload(),
    )

    assert response.status_code == 503

    assert response.json() == {
        "detail": (
            "Risk scoring is temporarily unavailable."
        )
    }

    assert (
        "Internal detail"
        not in response.text
    )


def test_api_endpoint_contains_no_duplicate_ml_logic() -> None:
    source = inspect.getsource(
        score_payment
    )

    assert (
        "score_capture_time_payment"
        in source
    )

    forbidden_implementation = {
        "predict_proba",
        "calculate_expected_costs",
        "evaluate_rules",
        "calibrate_probability",
        "model_bundle",
        "calibrator_bundle",
    }

    for value in forbidden_implementation:
        assert value not in source


def test_response_contains_no_automatic_execution(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "score_capture_time_payment",
        lambda payload: _fake_result(),
    )

    response = client.post(
        "/api/score",
        json=_request_payload(),
    )

    result = response.json()

    assert result["action_executed"] is False
    assert "refund_executed" not in result
    assert "message_sent" not in result
    assert "dispute_submitted" not in result