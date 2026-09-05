import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from src.razorpay_adapter import (
    RazorpayConfig,
    RazorpayNotFoundError,
    RazorpayOrder,
    RazorpayPayment,
)
from src.razorpay_api import (
    get_razorpay_adapter,
    get_razorpay_config,
    get_razorpay_service,
    router,
)
from src.razorpay_service import (
    RazorpayOrderResult,
)
from src.razorpay_store import (
    RazorpayIdempotencyConflictError,
)


class FakeService:
    def __init__(self) -> None:
        self.order_calls = 0
        self.conflict = False

    def create_order(
        self,
        intent,
        *,
        idempotency_key,
    ):
        self.order_calls += 1

        if self.conflict:
            raise (
                RazorpayIdempotencyConflictError(
                    "Internal conflict"
                )
            )

        return RazorpayOrderResult(
            delivery_mode="CREATED",
            checkout_key_id=(
                "rzp_test_public123"
            ),
            order_id="order_Test123",
            amount=intent.amount,
            currency="INR",
            receipt=intent.receipt,
            status="created",
            real_money_used=False,
            action_executed=False,
        )

class FakeAdapter:
    def __init__(self) -> None:
        self.not_found = False

    def fetch_order(
        self,
        order_id,
    ):
        if self.not_found:
            raise RazorpayNotFoundError(
                "Provider detail"
            )

        return RazorpayOrder(
            id=order_id,
            entity="order",
            amount=250000,
            amount_paid=0,
            amount_due=250000,
            currency="INR",
            receipt="radar_demo_001",
            status="created",
            attempts=0,
            notes={
                "private": "must-not-return",
            },
            created_at=1788540000,
        )

    def fetch_payment(
        self,
        payment_id,
    ):
        if self.not_found:
            raise RazorpayNotFoundError(
                "Provider detail"
            )

        return RazorpayPayment.model_validate(
            {
                "id": payment_id,
                "entity": "payment",
                "amount": 250000,
                "currency": "INR",
                "status": "captured",
                "order_id": "order_Test123",
                "method": "card",
                "captured": True,
                "created_at": 1788540100,
                "email": (
                    "forbidden@example.com"
                ),
                "contact": "+919999999999",
                "customer_id": "cust_forbidden",
            }
        )


def _test_client():
    application = FastAPI()
    application.include_router(router)

    service = FakeService()
    adapter = FakeAdapter()

    config = RazorpayConfig(
        enabled=True,
        test_mode=True,
        key_id="rzp_test_public123",
        key_secret=SecretStr(
            "never_return_this_secret"
        ),
    )

    application.dependency_overrides[
        get_razorpay_service
    ] = lambda: service

    application.dependency_overrides[
        get_razorpay_adapter
    ] = lambda: adapter

    application.dependency_overrides[
        get_razorpay_config
    ] = lambda: config

    return (
        TestClient(application),
        service,
        adapter,
    )


def test_disabled_dependency_returns_safe_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "RAZORPAY_INTEGRATION_ENABLED",
        "false",
    )
    monkeypatch.setenv(
        "RAZORPAY_KEY_ID",
        "",
    )
    monkeypatch.setenv(
        "RAZORPAY_KEY_SECRET",
        "",
    )

    get_razorpay_service.cache_clear()
    get_razorpay_adapter.cache_clear()
    get_razorpay_config.cache_clear()

    application = FastAPI()
    application.include_router(router)

    try:
        with TestClient(
            application,
            raise_server_exceptions=False,
        ) as client:
            response = client.post(
                "/api/razorpay-test/orders",
                headers={
                    "Idempotency-Key": (
                        "disabled_test_001"
                    )
                },
                json={
                    "amount": 100,
                    "currency": "INR",
                    "receipt": "disabled_test",
                },
            )
    finally:
        get_razorpay_service.cache_clear()
        get_razorpay_adapter.cache_clear()
        get_razorpay_config.cache_clear()

    assert response.status_code == 503
    assert response.json() == {
        "detail": (
            "Razorpay Test Mode integration is "
            "unavailable."
        )
    }


def test_status_contains_no_secret() -> None:
    client, _, _ = _test_client()

    response = client.get(
        "/api/razorpay-test/status"
    )

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "ready"
    assert result["test_mode"] is True
    assert result["credentials_configured"] is True
    assert result["automatic_financial_actions"] is False

    assert "key_secret" not in response.text
    assert "never_return" not in response.text


def test_order_requires_idempotency_header() -> None:
    client, _, _ = _test_client()

    response = client.post(
        "/api/razorpay-test/orders",
        json={
            "amount": 250000,
            "receipt": "radar_demo_001",
        },
    )

    assert response.status_code == 422


def test_openapi_excludes_direct_payment_scoring_route() -> None:
    application = FastAPI()
    application.include_router(router)

    assert (
        "/api/razorpay-test/payments/{payment_id}/score"
        not in application.openapi()["paths"]
    )


def test_create_order_is_bounded() -> None:
    client, service, _ = _test_client()

    response = client.post(
        "/api/razorpay-test/orders",
        headers={
            "Idempotency-Key": "request_0001"
        },
        json={
            "amount": 250000,
            "currency": "INR",
            "receipt": "radar_demo_001",
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["source"] == "RAZORPAY_TEST_MODE"
    assert result["real_money_used"] is False
    assert result["action_executed"] is False
    assert result["order_id"] == "order_Test123"
    assert service.order_calls == 1

    assert "key_secret" not in response.text


def test_order_conflict_returns_409() -> None:
    client, service, _ = _test_client()

    service.conflict = True

    response = client.post(
        "/api/razorpay-test/orders",
        headers={
            "Idempotency-Key": "request_0001"
        },
        json={
            "amount": 250000,
            "receipt": "radar_demo_001",
        },
    )

    assert response.status_code == 409
    assert "Internal conflict" not in response.text


def test_fetch_order_drops_private_notes() -> None:
    client, _, _ = _test_client()

    response = client.get(
        "/api/razorpay-test/orders/order_Test123"
    )

    assert response.status_code == 200

    result = response.json()

    assert result["order_id"] == "order_Test123"
    assert "notes" not in result
    assert result["real_money_used"] is False


def test_fetch_payment_drops_identity() -> None:
    client, _, _ = _test_client()

    response = client.get(
        "/api/razorpay-test/payments/pay_Test123"
    )

    assert response.status_code == 200

    result = response.json()

    assert result["payment_id"] == "pay_Test123"
    assert result["action_executed"] is False

    assert "email" not in result
    assert "contact" not in result
    assert "customer_id" not in result
    assert "forbidden@example.com" not in response.text


def test_direct_payment_scoring_route_is_unavailable() -> None:
    client, _, _ = _test_client()

    response = client.post(
        "/api/razorpay-test/payments/pay_Test123/score",
        json={},
    )

    assert response.status_code == 404


def test_unknown_provider_resource_returns_404() -> None:
    client, _, adapter = _test_client()

    adapter.not_found = True

    response = client.get(
        "/api/razorpay-test/payments/pay_Unknown123"
    )

    assert response.status_code == 404

    assert response.json() == {
        "detail": (
            "Razorpay Test payment not found."
        )
    }


def test_invalid_provider_identifier_returns_422() -> None:
    client, _, _ = _test_client()

    response = client.get(
        "/api/razorpay-test/payments/not-valid"
    )

    assert response.status_code == 422
