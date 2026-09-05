from __future__ import annotations

from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.razorpay_api import (
    get_razorpay_service,
)
from src.razorpay_checkout import (
    RazorpayCheckoutVerificationResult,
)
from src.razorpay_service import (
    RazorpayPaymentRiskResult,
    RazorpayServiceAuthenticationError,
    RazorpayServiceUnavailableError,
    RazorpayVerifiedPaymentRiskResult,
)
from src.razorpay_store import (
    RazorpayIdempotencyConflictError,
)
from src.risk_scoring import RiskScoreResult


ORDER_ID = "order_TestOrder12345"
PAYMENT_ID = "pay_TestPayment12345"
SIGNATURE = "a" * 64


def request_body() -> dict:
    return {
        "checkout": {
            "razorpay_payment_id": PAYMENT_ID,
            "razorpay_signature": SIGNATURE,
        },
        "context": {
            "feature_as_of": (
                "2026-09-05T00:00:00Z"
            ),
            "card_network": "Visa",
            "product_category": "electronics",
            "is_digital_good": False,
            "descriptor_clarity_score": 0.8,
            "phone_verified": True,
            "email_verified": True,
            "account_age_days": 365,
            "has_prior_order": 1,
            "total_prior_orders": 5,
            "prior_disputes_count": 0,
            "days_since_last_order": 30,
            "txns_last_1h": 0,
            "txns_last_24h": 0,
            "txns_last_7d": 1,
            "amount_last_24h_paise": 0,
            "device_is_new": False,
            "ip_country_matches_billing": True,
            "ip_is_proxy_or_vpn": False,
            "cvv_result": "match",
            "threeds_status": "authenticated",
            "threeds_liability_shift": True,
            "billing_shipping_distance_km": 10,
            "is_duplicate_payment": False,
            "cancelled_subscription_billed": False,
        },
    }


def successful_result() -> (
    RazorpayVerifiedPaymentRiskResult
):
    verification = (
        RazorpayCheckoutVerificationResult(
            verified=True,
            razorpay_order_id=ORDER_ID,
            razorpay_payment_id=PAYMENT_ID,
            algorithm="HMAC_SHA256",
            source="RAZORPAY_TEST_MODE",
            synthetic_evaluation_affected=False,
            financial_action_executed=False,
        )
    )

    score = RiskScoreResult.model_construct(
        raw_probability=0.1,
        calibrated_probability=0.08,
        risk_percentage=8.0,
        model_recommended_action="MONITOR",
        recommended_action="MONITOR",
        decision_source="COST_OPTIMIZED_MODEL",
        rules={},
        risk_band="HIGH",
        explanation={},
        expected_costs_rupees={
            "MONITOR": 10.0,
            "PREPARE_EVIDENCE": 20.0,
            "MANUAL_REVIEW": 30.0,
            "RECOMMEND_REFUND": 100.0,
        },
        model_version="0.1.0",
        calibration_version="0.1.0",
        calibration_method="isotonic",
        requires_human_approval=False,
        action_executed=False,
        disclosure="Synthetic model decision support.",
    )

    risk_result = (
        RazorpayPaymentRiskResult.model_construct(
            delivery_mode="CALCULATED",
            razorpay_payment_id=PAYMENT_ID,
            razorpay_order_id=ORDER_ID,
            provider_status="authorized",
            provider_method="card",
            normalization_version=(
                "razorpay-normalizer-v1"
            ),
            score=score,
            source="RAZORPAY_TEST_MODE",
            synthetic_evaluation_affected=False,
            human_approval_required=True,
            financial_action_executed=False,
        )
    )

    return RazorpayVerifiedPaymentRiskResult(
        verification=verification,
        risk_result=risk_result,
        source="RAZORPAY_TEST_MODE",
        signature_required_before_scoring=True,
        synthetic_evaluation_affected=False,
        financial_action_executed=False,
    )


class FakeService:
    def __init__(
        self,
        error: Exception | None = None,
    ) -> None:
        self.error = error
        self.calls: list[dict] = []

    def verify_and_score_checkout(
        self,
        **kwargs: object,
    ) -> RazorpayVerifiedPaymentRiskResult:
        self.calls.append(kwargs)

        if self.error is not None:
            raise self.error

        return successful_result()


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def override_service(
    service: FakeService,
) -> None:
    app.dependency_overrides[
        get_razorpay_service
    ] = lambda: service


def endpoint() -> str:
    return (
        f"/api/razorpay-test/orders/"
        f"{ORDER_ID}/verify-and-score"
    )


def test_verified_payment_is_scored(
    client: TestClient,
) -> None:
    service = FakeService()
    override_service(service)

    response = client.post(
        endpoint(),
        json=request_body(),
    )

    assert response.status_code == 200

    payload = response.json()

    assert (
        payload["verification"]["verified"]
        is True
    )

    assert (
        payload["risk_result"]["score"][
            "calibrated_probability"
        ]
        == 0.08
    )

    assert (
        payload["risk_result"]["score"][
            "recommended_action"
        ]
        == "MONITOR"
    )

    assert (
        payload[
            "signature_required_before_scoring"
        ]
        is True
    )

    assert (
        payload["financial_action_executed"]
        is False
    )

    serialized = response.text.lower()
    assert "dispute_created" not in serialized
    assert "refund_executed" not in serialized
    assert "message_sent" not in serialized
    assert "real issuer dispute" not in serialized

    assert len(service.calls) == 1


def test_signature_failure_returns_401(
    client: TestClient,
) -> None:
    override_service(
        FakeService(
            RazorpayServiceAuthenticationError(
                "Sensitive signature failure"
            )
        )
    )

    response = client.post(
        endpoint(),
        json=request_body(),
    )

    assert response.status_code == 401

    assert (
        "Sensitive signature failure"
        not in response.text
    )


def test_context_conflict_returns_409(
    client: TestClient,
) -> None:
    override_service(
        FakeService(
            RazorpayIdempotencyConflictError(
                "Sensitive context conflict"
            )
        )
    )

    response = client.post(
        endpoint(),
        json=request_body(),
    )

    assert response.status_code == 409


def test_provider_failure_returns_503(
    client: TestClient,
) -> None:
    override_service(
        FakeService(
            RazorpayServiceUnavailableError(
                "Sensitive provider failure"
            )
        )
    )

    response = client.post(
        endpoint(),
        json=request_body(),
    )

    assert response.status_code == 503

    assert (
        "Sensitive provider failure"
        not in response.text
    )


def test_forbidden_fields_are_rejected(
    client: TestClient,
) -> None:
    service = FakeService()
    override_service(service)

    body = request_body()

    body["context"]["customer_id"] = (
        "cust_forbidden"
    )

    body["context"][
        "chargeback_within_120d"
    ] = 1

    response = client.post(
        endpoint(),
        json=body,
    )

    assert response.status_code == 422
    assert service.calls == []
