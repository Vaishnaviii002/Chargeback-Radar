from __future__ import annotations

import hashlib
import hmac
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from src.razorpay_adapter import (
    RazorpayAdapterError,
)
from src.razorpay_service import (
    RazorpayRiskService,
    RazorpayServiceAuthenticationError,
    RazorpayServiceUnavailableError,
    RazorpayServiceValidationError,
)


ORDER_ID = "order_TestOrder12345"
PAYMENT_ID = "pay_TestPayment12345"
KEY_SECRET = "test_secret_for_checkout"


def make_signature(
    order_id: str = ORDER_ID,
    payment_id: str = PAYMENT_ID,
) -> str:
    payload = (
        f"{order_id}|{payment_id}"
    ).encode("utf-8")

    return hmac.new(
        KEY_SECRET.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()


class FakeAdapter:
    def __init__(
        self,
        *,
        order_id: str = ORDER_ID,
        payment_order_id: str = ORDER_ID,
        order_amount: int = 10_000,
        payment_amount: int = 10_000,
        currency: str = "INR",
        payment_method: str = "card",
        payment_status: str = "authorized",
        key_secret: SecretStr | None = None,
    ) -> None:
        self.config = SimpleNamespace(
            key_secret=(
                key_secret
                if key_secret is not None
                else SecretStr(KEY_SECRET)
            )
        )

        self.order = SimpleNamespace(
            id=order_id,
            amount=order_amount,
            currency=currency,
            status="attempted",
        )

        self.payment = SimpleNamespace(
            id=PAYMENT_ID,
            order_id=payment_order_id,
            amount=payment_amount,
            currency=currency,
            method=payment_method,
            status=payment_status,
        )

        self.order_fetches = 0
        self.payment_fetches = 0

    def fetch_order(
        self,
        order_id: str,
    ) -> object:
        self.order_fetches += 1
        return self.order

    def fetch_payment(
        self,
        payment_id: str,
    ) -> object:
        self.payment_fetches += 1
        return self.payment


def make_service(
    adapter: FakeAdapter,
) -> RazorpayRiskService:
    return RazorpayRiskService(
        adapter=adapter,
        store=SimpleNamespace(),
        scorer=lambda _: None,
    )


def test_valid_checkout_is_bound_to_provider_records() -> None:
    adapter = FakeAdapter()
    service = make_service(adapter)

    result = service.verify_checkout_payment(
        server_order_id=ORDER_ID,
        razorpay_payment_id=PAYMENT_ID,
        razorpay_signature=make_signature(),
    )

    assert result.verified is True
    assert result.razorpay_order_id == ORDER_ID
    assert result.razorpay_payment_id == PAYMENT_ID
    assert adapter.order_fetches == 1
    assert adapter.payment_fetches == 1
    assert (
        result.financial_action_executed
        is False
    )


def test_invalid_signature_fails_before_provider_fetch() -> None:
    adapter = FakeAdapter()
    service = make_service(adapter)

    with pytest.raises(
        RazorpayServiceAuthenticationError
    ):
        service.verify_checkout_payment(
            server_order_id=ORDER_ID,
            razorpay_payment_id=PAYMENT_ID,
            razorpay_signature="a" * 64,
        )

    assert adapter.order_fetches == 0
    assert adapter.payment_fetches == 0


def test_payment_must_belong_to_verified_order() -> None:
    adapter = FakeAdapter(
        payment_order_id=(
            "order_DifferentOrder123"
        )
    )

    with pytest.raises(
        RazorpayServiceAuthenticationError
    ):
        make_service(
            adapter
        ).verify_checkout_payment(
            server_order_id=ORDER_ID,
            razorpay_payment_id=PAYMENT_ID,
            razorpay_signature=make_signature(),
        )


def test_payment_amount_must_match_order() -> None:
    adapter = FakeAdapter(
        payment_amount=20_000
    )

    with pytest.raises(
        RazorpayServiceValidationError,
        match="amount",
    ):
        make_service(
            adapter
        ).verify_checkout_payment(
            server_order_id=ORDER_ID,
            razorpay_payment_id=PAYMENT_ID,
            razorpay_signature=make_signature(),
        )


def test_only_card_payment_is_accepted() -> None:
    adapter = FakeAdapter(
        payment_method="upi"
    )

    with pytest.raises(
        RazorpayServiceValidationError,
        match="card",
    ):
        make_service(
            adapter
        ).verify_checkout_payment(
            server_order_id=ORDER_ID,
            razorpay_payment_id=PAYMENT_ID,
            razorpay_signature=make_signature(),
        )


@pytest.mark.parametrize(
    "payment_status",
    [
        "created",
        "failed",
        "refunded",
    ],
)
def test_unsupported_payment_status_is_rejected(
    payment_status: str,
) -> None:
    adapter = FakeAdapter(
        payment_status=payment_status
    )

    with pytest.raises(
        RazorpayServiceValidationError,
        match="authorized or captured",
    ):
        make_service(
            adapter
        ).verify_checkout_payment(
            server_order_id=ORDER_ID,
            razorpay_payment_id=PAYMENT_ID,
            razorpay_signature=make_signature(),
        )


def test_missing_secret_fails_closed() -> None:
    adapter = FakeAdapter()
    adapter.config.key_secret = None

    with pytest.raises(
        RazorpayServiceUnavailableError
    ):
        make_service(
            adapter
        ).verify_checkout_payment(
            server_order_id=ORDER_ID,
            razorpay_payment_id=PAYMENT_ID,
            razorpay_signature=make_signature(),
        )

    assert adapter.order_fetches == 0
    assert adapter.payment_fetches == 0


def test_provider_failure_is_safe() -> None:
    adapter = FakeAdapter()

    def fail_fetch(
        order_id: str,
    ) -> object:
        raise RazorpayAdapterError(
            "Sensitive provider failure"
        )

    adapter.fetch_order = fail_fetch

    with pytest.raises(
        RazorpayServiceUnavailableError
    ):
        make_service(
            adapter
        ).verify_checkout_payment(
            server_order_id=ORDER_ID,
            razorpay_payment_id=PAYMENT_ID,
            razorpay_signature=make_signature(),
        )