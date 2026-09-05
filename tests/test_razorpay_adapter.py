import json

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from src.razorpay_adapter import (
    RAZORPAY_API_BASE_URL,
    RazorpayAdapter,
    RazorpayConfigurationError,
    RazorpayDisabledError,
    RazorpayNotFoundError,
    RazorpayOrderCreateRequest,
    RazorpayResponseError,
    RazorpayUnavailableError,
    RazorpayConfig,
)


TEST_SECRET = "test_secret_value_123"


def _config() -> RazorpayConfig:
    return RazorpayConfig(
        enabled=True,
        test_mode=True,
        key_id="rzp_test_unit123",
        key_secret=SecretStr(TEST_SECRET),
        timeout_seconds=5,
    )


def _order_payload() -> dict:
    return {
        "id": "order_Test123",
        "entity": "order",
        "amount": 250000,
        "amount_paid": 0,
        "amount_due": 250000,
        "currency": "INR",
        "receipt": "chargeback_radar_001",
        "status": "created",
        "attempts": 0,
        "notes": {
            "source": "chargeback_radar",
        },
        "created_at": 1788540000,
    }


def _payment_payload() -> dict:
    return {
        "id": "pay_Test123",
        "entity": "payment",
        "amount": 250000,
        "currency": "INR",
        "status": "captured",
        "order_id": "order_Test123",
        "method": "card",
        "captured": True,
        "description": "Test payment",
        "bank": None,
        "wallet": None,
        "email": "must-not-be-retained@example.com",
        "contact": "+919999999999",
        "created_at": 1788540100,
    }


def _adapter(
    handler,
) -> tuple[RazorpayAdapter, httpx.Client]:
    transport = httpx.MockTransport(handler)

    client = httpx.Client(
        transport=transport,
        base_url=RAZORPAY_API_BASE_URL,
    )

    return (
        RazorpayAdapter(
            config=_config(),
            client=client,
        ),
        client,
    )


def test_config_loads_test_credentials_from_env(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "RAZORPAY_INTEGRATION_ENABLED",
        "true",
    )
    monkeypatch.setenv(
        "RAZORPAY_TEST_MODE",
        "true",
    )
    monkeypatch.setenv(
        "RAZORPAY_KEY_ID",
        "rzp_test_environment",
    )
    monkeypatch.setenv(
        "RAZORPAY_KEY_SECRET",
        TEST_SECRET,
    )

    config = RazorpayConfig.from_env()

    key_id, secret = (
        config.require_test_credentials()
    )

    assert key_id == "rzp_test_environment"
    assert secret == TEST_SECRET


def test_secret_is_masked_in_config_repr() -> None:
    representation = repr(_config())

    assert TEST_SECRET not in representation
    assert "**********" in representation


def test_disabled_integration_fails_closed() -> None:
    config = RazorpayConfig(
        enabled=False,
    )

    with pytest.raises(
        RazorpayDisabledError
    ):
        RazorpayAdapter(config=config)


def test_live_key_is_rejected() -> None:
    config = RazorpayConfig(
        enabled=True,
        test_mode=True,
        key_id="rzp_live_forbidden",
        key_secret=SecretStr(TEST_SECRET),
    )

    with pytest.raises(
        RazorpayConfigurationError
    ):
        RazorpayAdapter(config=config)


def test_test_mode_cannot_be_disabled() -> None:
    config = RazorpayConfig(
        enabled=True,
        test_mode=False,
        key_id="rzp_test_unit123",
        key_secret=SecretStr(TEST_SECRET),
    )

    with pytest.raises(
        RazorpayConfigurationError
    ):
        RazorpayAdapter(config=config)


def test_order_request_is_strict() -> None:
    with pytest.raises(ValidationError):
        RazorpayOrderCreateRequest.model_validate(
            {
                "amount": 250000,
                "currency": "INR",
                "receipt": "receipt_001",
                "unexpected": True,
            }
        )


def test_order_request_rejects_invalid_amount() -> None:
    with pytest.raises(ValidationError):
        RazorpayOrderCreateRequest(
            amount=0,
            receipt="receipt_001",
        )


def test_create_order_uses_expected_payload_and_auth() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/orders"

        payload = json.loads(
            request.content.decode("utf-8")
        )

        assert payload == {
            "amount": 250000,
            "currency": "INR",
            "receipt": "chargeback_radar_001",
            "notes": {
                "source": "chargeback_radar",
            },
        }

        assert request.headers[
            "authorization"
        ].startswith("Basic ")

        return httpx.Response(
            200,
            json=_order_payload(),
        )

    adapter, client = _adapter(handler)

    try:
        order = adapter.create_order(
            RazorpayOrderCreateRequest(
                amount=250000,
                receipt="chargeback_radar_001",
                notes={
                    "source": "chargeback_radar",
                },
            )
        )
    finally:
        client.close()

    assert order.id == "order_Test123"
    assert order.amount == 250000
    assert order.status == "created"


def test_fetch_order() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        assert request.method == "GET"
        assert (
            request.url.path
            == "/v1/orders/order_Test123"
        )

        return httpx.Response(
            200,
            json=_order_payload(),
        )

    adapter, client = _adapter(handler)

    try:
        order = adapter.fetch_order(
            "order_Test123"
        )
    finally:
        client.close()

    assert order.receipt == "chargeback_radar_001"


def test_fetch_payment_drops_identity_fields() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        assert request.method == "GET"
        assert (
            request.url.path
            == "/v1/payments/pay_Test123"
        )

        return httpx.Response(
            200,
            json=_payment_payload(),
        )

    adapter, client = _adapter(handler)

    try:
        payment = adapter.fetch_payment(
            "pay_Test123"
        )
    finally:
        client.close()

    serialized = payment.model_dump()

    assert payment.status == "captured"
    assert payment.method == "card"
    assert "email" not in serialized
    assert "contact" not in serialized
    assert TEST_SECRET not in str(serialized)


def test_invalid_provider_id_is_rejected_before_request() -> None:
    adapter, client = _adapter(
        lambda request: pytest.fail(
            "HTTP request should not be made."
        )
    )

    try:
        with pytest.raises(ValueError):
            adapter.fetch_payment(
                "../../payments"
            )
    finally:
        client.close()


def test_not_found_is_normalized() -> None:
    adapter, client = _adapter(
        lambda request: httpx.Response(
            404,
            json={"error": "not found"},
        )
    )

    try:
        with pytest.raises(
            RazorpayNotFoundError
        ):
            adapter.fetch_order(
                "order_Unknown123"
            )
    finally:
        client.close()


def test_provider_error_does_not_expose_secrets() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            401,
            headers={
                "x-razorpay-request-id": "req_test_001",
            },
            json={
                "error": {
                    "description": TEST_SECRET,
                }
            },
        )

    adapter, client = _adapter(handler)

    try:
        with pytest.raises(
            RazorpayUnavailableError
        ) as captured:
            adapter.fetch_order(
                "order_Test123"
            )
    finally:
        client.close()

    message = str(captured.value)

    assert TEST_SECRET not in message
    assert "401" in message
    assert "req_test_001" in message


def test_invalid_json_is_rejected() -> None:
    adapter, client = _adapter(
        lambda request: httpx.Response(
            200,
            content=b"not-json",
        )
    )

    try:
        with pytest.raises(
            RazorpayResponseError
        ):
            adapter.fetch_order(
                "order_Test123"
            )
    finally:
        client.close()


def test_invalid_provider_object_is_rejected() -> None:
    adapter, client = _adapter(
        lambda request: httpx.Response(
            200,
            json={"id": "invalid"},
        )
    )

    try:
        with pytest.raises(
            RazorpayResponseError
        ):
            adapter.fetch_order(
                "order_Test123"
            )
    finally:
        client.close()



def test_router_is_mounted_in_main_application() -> None:
    from src.api import app as main_app

    paths = set(
        main_app.openapi()["paths"]
    )

    expected = {
        "/api/razorpay-test/status",
        "/api/razorpay-test/webhooks",
        "/api/razorpay-test/orders",
        "/api/razorpay-test/orders/{order_id}",
        (
            "/api/razorpay-test/orders/"
            "{order_id}/verify-checkout"
        ),
        (
            "/api/razorpay-test/orders/"
            "{order_id}/verify-and-score"
        ),
        "/api/razorpay-test/payments/{payment_id}",
    }

    assert expected.issubset(paths)
    assert (
        "/api/razorpay-test/payments/{payment_id}/score"
        not in paths
    )
