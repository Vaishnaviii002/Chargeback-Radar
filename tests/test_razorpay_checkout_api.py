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
    RazorpayServiceAuthenticationError,
    RazorpayServiceUnavailableError,
    RazorpayServiceValidationError,
)


ORDER_ID = "order_TestOrder12345"
PAYMENT_ID = "pay_TestPayment12345"
SIGNATURE = "a" * 64


class FakeVerificationService:
    def __init__(
        self,
        *,
        error: Exception | None = None,
    ) -> None:
        self.error = error
        self.calls: list[dict[str, str]] = []

    def verify_checkout_payment(
        self,
        *,
        server_order_id: str,
        razorpay_payment_id: str,
        razorpay_signature: str,
    ) -> RazorpayCheckoutVerificationResult:
        self.calls.append(
            {
                "server_order_id": server_order_id,
                "razorpay_payment_id": (
                    razorpay_payment_id
                ),
                "razorpay_signature": (
                    razorpay_signature
                ),
            }
        )

        if self.error is not None:
            raise self.error

        return RazorpayCheckoutVerificationResult(
            verified=True,
            razorpay_order_id=server_order_id,
            razorpay_payment_id=(
                razorpay_payment_id
            ),
            algorithm="HMAC_SHA256",
            source="RAZORPAY_TEST_MODE",
            synthetic_evaluation_affected=False,
            financial_action_executed=False,
        )


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def override_service(
    service: FakeVerificationService,
) -> None:
    app.dependency_overrides[
        get_razorpay_service
    ] = lambda: service


def test_valid_checkout_verification(
    client: TestClient,
) -> None:
    service = FakeVerificationService()
    override_service(service)

    response = client.post(
        (
            f"/api/razorpay-test/orders/"
            f"{ORDER_ID}/verify-checkout"
        ),
        json={
            "razorpay_payment_id": PAYMENT_ID,
            "razorpay_signature": SIGNATURE,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["verified"] is True
    assert (
        payload["razorpay_order_id"]
        == ORDER_ID
    )
    assert (
        payload["razorpay_payment_id"]
        == PAYMENT_ID
    )
    assert (
        payload["source"]
        == "RAZORPAY_TEST_MODE"
    )
    assert (
        payload["financial_action_executed"]
        is False
    )

    serialized = response.text

    assert SIGNATURE not in serialized
    assert "key_secret" not in serialized
    assert "razorpay_signature" not in serialized

    assert service.calls == [
        {
            "server_order_id": ORDER_ID,
            "razorpay_payment_id": PAYMENT_ID,
            "razorpay_signature": SIGNATURE,
        }
    ]


def test_invalid_signature_returns_401(
    client: TestClient,
) -> None:
    override_service(
        FakeVerificationService(
            error=(
                RazorpayServiceAuthenticationError(
                    "Sensitive internal information"
                )
            )
        )
    )

    response = client.post(
        (
            f"/api/razorpay-test/orders/"
            f"{ORDER_ID}/verify-checkout"
        ),
        json={
            "razorpay_payment_id": PAYMENT_ID,
            "razorpay_signature": SIGNATURE,
        },
    )

    assert response.status_code == 401
    assert (
        response.json()["detail"]
        == (
            "Razorpay Checkout signature "
            "verification failed."
        )
    )
    assert (
        "Sensitive internal information"
        not in response.text
    )


def test_invalid_provider_binding_returns_422(
    client: TestClient,
) -> None:
    override_service(
        FakeVerificationService(
            error=RazorpayServiceValidationError(
                "Payment amount does not match."
            )
        )
    )

    response = client.post(
        (
            f"/api/razorpay-test/orders/"
            f"{ORDER_ID}/verify-checkout"
        ),
        json={
            "razorpay_payment_id": PAYMENT_ID,
            "razorpay_signature": SIGNATURE,
        },
    )

    assert response.status_code == 422


def test_provider_failure_returns_503(
    client: TestClient,
) -> None:
    override_service(
        FakeVerificationService(
            error=RazorpayServiceUnavailableError(
                "Sensitive provider failure"
            )
        )
    )

    response = client.post(
        (
            f"/api/razorpay-test/orders/"
            f"{ORDER_ID}/verify-checkout"
        ),
        json={
            "razorpay_payment_id": PAYMENT_ID,
            "razorpay_signature": SIGNATURE,
        },
    )

    assert response.status_code == 503
    assert (
        "Sensitive provider failure"
        not in response.text
    )


def test_extra_checkout_fields_are_rejected(
    client: TestClient,
) -> None:
    service = FakeVerificationService()
    override_service(service)

    response = client.post(
        (
            f"/api/razorpay-test/orders/"
            f"{ORDER_ID}/verify-checkout"
        ),
        json={
            "razorpay_payment_id": PAYMENT_ID,
            "razorpay_signature": SIGNATURE,
            "customer_id": "cust_forbidden",
            "chargeback_within_120d": 1,
        },
    )

    assert response.status_code == 422
    assert service.calls == []


def test_invalid_signature_shape_is_rejected(
    client: TestClient,
) -> None:
    service = FakeVerificationService()
    override_service(service)

    response = client.post(
        (
            f"/api/razorpay-test/orders/"
            f"{ORDER_ID}/verify-checkout"
        ),
        json={
            "razorpay_payment_id": PAYMENT_ID,
            "razorpay_signature": "invalid",
        },
    )

    assert response.status_code == 422
    assert service.calls == []