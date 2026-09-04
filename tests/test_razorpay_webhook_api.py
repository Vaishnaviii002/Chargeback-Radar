from __future__ import annotations

from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.razorpay_api import (
    get_razorpay_webhook_service,
)
from src.razorpay_store import (
    RazorpayIdempotencyConflictError,
    RazorpayStoreError,
)
from src.razorpay_webhook import (
    MAX_WEBHOOK_BODY_BYTES,
    RazorpayVerifiedWebhook,
    RazorpayWebhookPayloadError,
    RazorpayWebhookSignatureError,
)
from src.razorpay_webhook_service import (
    RazorpayWebhookDeliveryResult,
    RazorpayWebhookDisabledError,
)


EVENT_ID = "event_test_000001"
SIGNATURE = "a" * 64

RAW_BODY = (
    b'{"entity":"event",'
    b'"event":"payment.captured",'
    b'"created_at":1788550000,'
    b'"payload":{"payment":{"entity":{'
    b'"id":"pay_TestPayment12345",'
    b'"order_id":"order_TestOrder12345",'
    b'"email":"forbidden@example.com"'
    b"}}}}"
)


def delivery_result(
    delivery_mode: str = "PROCESSED",
) -> RazorpayWebhookDeliveryResult:
    event = RazorpayVerifiedWebhook(
        verified=True,
        event_id=EVENT_ID,
        event_type="payment.captured",
        created_at=1788550000,
        processing_decision="PROCESS",
        resource_kind="payment",
        resource_id=(
            "pay_TestPayment12345"
        ),
        payment_id="pay_TestPayment12345",
        order_id="order_TestOrder12345",
        signature_algorithm="HMAC_SHA256",
        source="RAZORPAY_WEBHOOK",
        financial_action_executed=False,
    )

    return RazorpayWebhookDeliveryResult(
        delivery_mode=delivery_mode,
        event=event,
        duplicate=(
            delivery_mode
            == "IDEMPOTENT_REPLAY"
        ),
        stored_status="PROCESSED",
        human_approval_required=True,
        financial_action_executed=False,
    )


class FakeWebhookService:
    def __init__(
        self,
        *,
        error: Exception | None = None,
        delivery_mode: str = "PROCESSED",
    ) -> None:
        self.error = error
        self.delivery_mode = delivery_mode
        self.calls: list[dict] = []

    def process(
        self,
        *,
        raw_body: bytes,
        signature: str,
        event_id: str,
    ) -> RazorpayWebhookDeliveryResult:
        self.calls.append(
            {
                "raw_body": raw_body,
                "signature": signature,
                "event_id": event_id,
            }
        )

        if self.error is not None:
            raise self.error

        return delivery_result(
            self.delivery_mode
        )


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def override_service(
    service: FakeWebhookService,
) -> None:
    app.dependency_overrides[
        get_razorpay_webhook_service
    ] = lambda: service


def webhook_headers() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-Razorpay-Signature": SIGNATURE,
        "X-Razorpay-Event-Id": EVENT_ID,
    }


def test_raw_body_reaches_service_unchanged(
    client: TestClient,
) -> None:
    service = FakeWebhookService()
    override_service(service)

    response = client.post(
        "/api/razorpay-test/webhooks",
        content=RAW_BODY,
        headers=webhook_headers(),
    )

    assert response.status_code == 200
    assert len(service.calls) == 1

    call = service.calls[0]

    assert call["raw_body"] == RAW_BODY
    assert call["signature"] == SIGNATURE
    assert call["event_id"] == EVENT_ID

    payload = response.json()

    assert (
        payload["delivery_mode"]
        == "PROCESSED"
    )

    assert (
        payload["event"]["payment_id"]
        == "pay_TestPayment12345"
    )

    assert (
        payload["financial_action_executed"]
        is False
    )

    assert (
        "forbidden@example.com"
        not in response.text
    )

    assert SIGNATURE not in response.text


def test_oversized_body_is_rejected_before_service(
    client: TestClient,
) -> None:
    service = FakeWebhookService()
    override_service(service)

    response = client.post(
        "/api/razorpay-test/webhooks",
        content=(
            b"x" * (MAX_WEBHOOK_BODY_BYTES + 1)
        ),
        headers=webhook_headers(),
    )

    assert response.status_code == 400
    assert service.calls == []
    assert response.json() == {
        "detail": (
            "Razorpay webhook payload is invalid."
        )
    }


def test_duplicate_event_returns_replay(
    client: TestClient,
) -> None:
    service = FakeWebhookService(
        delivery_mode="IDEMPOTENT_REPLAY"
    )

    override_service(service)

    response = client.post(
        "/api/razorpay-test/webhooks",
        content=RAW_BODY,
        headers=webhook_headers(),
    )

    assert response.status_code == 200

    assert (
        response.json()["delivery_mode"]
        == "IDEMPOTENT_REPLAY"
    )

    assert response.json()["duplicate"] is True


def test_invalid_signature_returns_401(
    client: TestClient,
) -> None:
    override_service(
        FakeWebhookService(
            error=RazorpayWebhookSignatureError(
                "Sensitive signature failure"
            )
        )
    )

    response = client.post(
        "/api/razorpay-test/webhooks",
        content=RAW_BODY,
        headers=webhook_headers(),
    )

    assert response.status_code == 401

    assert (
        "Sensitive signature failure"
        not in response.text
    )


def test_invalid_payload_returns_400(
    client: TestClient,
) -> None:
    override_service(
        FakeWebhookService(
            error=RazorpayWebhookPayloadError(
                "Sensitive payload failure"
            )
        )
    )

    response = client.post(
        "/api/razorpay-test/webhooks",
        content=RAW_BODY,
        headers=webhook_headers(),
    )

    assert response.status_code == 400

    assert (
        "Sensitive payload failure"
        not in response.text
    )


def test_event_id_conflict_returns_409(
    client: TestClient,
) -> None:
    override_service(
        FakeWebhookService(
            error=(
                RazorpayIdempotencyConflictError(
                    "Sensitive conflict"
                )
            )
        )
    )

    response = client.post(
        "/api/razorpay-test/webhooks",
        content=RAW_BODY,
        headers=webhook_headers(),
    )

    assert response.status_code == 409


@pytest.mark.parametrize(
    "error",
    [
        RazorpayWebhookDisabledError(
            "Webhook disabled"
        ),
        RazorpayStoreError(
            "Sensitive database failure"
        ),
    ],
)
def test_unavailable_webhook_returns_503(
    client: TestClient,
    error: Exception,
) -> None:
    override_service(
        FakeWebhookService(
            error=error
        )
    )

    response = client.post(
        "/api/razorpay-test/webhooks",
        content=RAW_BODY,
        headers=webhook_headers(),
    )

    assert response.status_code == 503

    assert str(error) not in response.text


def test_required_headers_are_enforced(
    client: TestClient,
) -> None:
    service = FakeWebhookService()
    override_service(service)

    response = client.post(
        "/api/razorpay-test/webhooks",
        content=RAW_BODY,
        headers={
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 422
    assert service.calls == []
