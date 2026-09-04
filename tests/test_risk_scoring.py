from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api import app
from src.decide import ACTIONS
from src.razorpay_normalizer import (
    CaptureTimeScoringPayload,
)
from src.risk_scoring import (
    load_model_bundle,
    score_capture_time_payment,
)


UTC = timezone.utc
client = TestClient(app)


def _payload() -> CaptureTimeScoringPayload:
    return CaptureTimeScoringPayload(
        created_at=datetime(
            2025,
            12,
            1,
            12,
            30,
            tzinfo=UTC,
        ),
        amount_paise=250000,
        card_network="Visa",
        product_category="electronics",
        is_digital_good=False,
        descriptor_clarity_score=0.82,
        phone_verified=True,
        email_verified=True,
        account_age_days=420,
        has_prior_order=1,
        total_prior_orders=8,
        prior_disputes_count=0,
        days_since_last_order=20,
        txns_last_1h=0,
        txns_last_24h=1,
        txns_last_7d=2,
        amount_last_24h_paise=250000,
        device_is_new=False,
        ip_country_matches_billing=True,
        ip_is_proxy_or_vpn=False,
        cvv_result="match",
        threeds_status="authenticated",
        threeds_liability_shift=True,
        billing_shipping_distance_km=12,
        is_duplicate_payment=False,
        cancelled_subscription_billed=False,
    )


def test_production_scoring_is_deterministic() -> None:
    first = score_capture_time_payment(
        _payload()
    )

    second = score_capture_time_payment(
        _payload()
    )

    assert first.model_dump() == second.model_dump()


def test_probability_and_risk_are_valid() -> None:
    result = score_capture_time_payment(
        _payload()
    )

    assert 0 <= result.raw_probability <= 1

    assert (
        0
        <= result.calibrated_probability
        <= 1
    )

    assert result.risk_percentage == round(
        result.calibrated_probability
        * 100,
        3,
    )

    assert result.risk_band in {
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    }


def test_cost_policy_contains_every_action() -> None:
    result = score_capture_time_payment(
        _payload()
    )

    assert set(
        result.expected_costs_rupees
    ) == set(ACTIONS)

    assert (
        result.model_recommended_action
        in ACTIONS
    )

    assert result.recommended_action in ACTIONS


def test_result_never_executes_an_action() -> None:
    result = score_capture_time_payment(
        _payload()
    )

    assert result.action_executed is False

    expected_approval = (
        result.recommended_action
        in {
            "MANUAL_REVIEW",
            "RECOMMEND_REFUND",
        }
    )

    assert (
        result.requires_human_approval
        is expected_approval
    )


def test_model_bundle_contains_only_safe_features() -> None:
    model_features = set(
        load_model_bundle()[
            "model_features"
        ]
    )

    forbidden = {
        "payment_id",
        "order_id",
        "customer_id",
        "email",
        "contact",
        "chargeback_within_120d",
        "target",
        "true_fraud",
        "dispute_status",
        "dispute_reason",
        "refund_status",
        "amount_refunded",
    }

    assert forbidden.isdisjoint(
        model_features
    )


def test_payload_rejects_sensitive_extra_fields() -> None:
    values = _payload().model_dump()

    values["customer_id"] = (
        "cust_forbidden"
    )

    with pytest.raises(ValidationError):
        CaptureTimeScoringPayload.model_validate(
            values
        )


def test_reusable_service_matches_existing_api() -> None:
    payload = _payload()

    service_result = (
        score_capture_time_payment(
            payload
        )
    )

    response = client.post(
        "/api/score",
        json=payload.model_dump(
            mode="json"
        ),
    )

    assert response.status_code == 200

    api_result = response.json()

    assert api_result[
        "raw_probability"
    ] == pytest.approx(
        service_result.raw_probability,
        abs=1e-12,
    )

    assert api_result[
        "calibrated_probability"
    ] == pytest.approx(
        service_result.calibrated_probability,
        abs=1e-12,
    )

    assert api_result[
        "recommended_action"
    ] == service_result.recommended_action

    assert api_result[
        "decision_source"
    ] == service_result.decision_source

    assert api_result[
        "expected_costs_rupees"
    ] == pytest.approx(
        service_result.expected_costs_rupees
    )

    assert api_result[
        "model_version"
    ] == service_result.model_version

    assert api_result[
        "calibration_method"
    ] == service_result.calibration_method


def test_disclosure_is_honest() -> None:
    disclosure = (
        score_capture_time_payment(
            _payload()
        ).disclosure
    )

    assert "synthetic data" in disclosure
    assert "decision-support" in disclosure
    assert "no financial action" in disclosure