from __future__ import annotations

import hashlib
import hmac

import pytest
from pydantic import (
    SecretStr,
    ValidationError,
)

from src.razorpay_checkout import (
    RazorpayCheckoutInputError,
    RazorpayCheckoutSignatureError,
    RazorpayCheckoutVerificationRequest,
    verify_checkout_signature,
)


ORDER_ID = "order_TestOrder12345"
PAYMENT_ID = "pay_TestPayment12345"
KEY_SECRET = "test_secret_for_checkout"


def make_signature(
    order_id: str = ORDER_ID,
    payment_id: str = PAYMENT_ID,
    secret: str = KEY_SECRET,
) -> str:
    payload = (
        f"{order_id}|{payment_id}"
    ).encode("utf-8")

    return hmac.new(
        secret.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()


def test_valid_checkout_signature() -> None:
    result = verify_checkout_signature(
        server_order_id=ORDER_ID,
        razorpay_payment_id=PAYMENT_ID,
        razorpay_signature=make_signature(),
        key_secret=KEY_SECRET,
    )

    assert result.verified is True
    assert result.razorpay_order_id == ORDER_ID
    assert (
        result.razorpay_payment_id
        == PAYMENT_ID
    )
    assert result.algorithm == "HMAC_SHA256"
    assert result.source == "RAZORPAY_TEST_MODE"
    assert (
        result.synthetic_evaluation_affected
        is False
    )
    assert (
        result.financial_action_executed
        is False
    )


def test_secret_str_is_supported() -> None:
    result = verify_checkout_signature(
        server_order_id=ORDER_ID,
        razorpay_payment_id=PAYMENT_ID,
        razorpay_signature=make_signature(),
        key_secret=SecretStr(KEY_SECRET),
    )

    assert result.verified is True


def test_tampered_order_id_is_rejected() -> None:
    with pytest.raises(
        RazorpayCheckoutSignatureError
    ):
        verify_checkout_signature(
            server_order_id=(
                "order_DifferentOrder123"
            ),
            razorpay_payment_id=PAYMENT_ID,
            razorpay_signature=make_signature(),
            key_secret=KEY_SECRET,
        )


def test_tampered_payment_id_is_rejected() -> None:
    with pytest.raises(
        RazorpayCheckoutSignatureError
    ):
        verify_checkout_signature(
            server_order_id=ORDER_ID,
            razorpay_payment_id=(
                "pay_DifferentPayment123"
            ),
            razorpay_signature=make_signature(),
            key_secret=KEY_SECRET,
        )


def test_wrong_secret_is_rejected() -> None:
    with pytest.raises(
        RazorpayCheckoutSignatureError
    ):
        verify_checkout_signature(
            server_order_id=ORDER_ID,
            razorpay_payment_id=PAYMENT_ID,
            razorpay_signature=make_signature(),
            key_secret="different_secret_value",
        )


@pytest.mark.parametrize(
    "order_id",
    [
        "",
        "not_an_order",
        "order_bad-value",
        "../order_TestOrder12345",
    ],
)
def test_invalid_order_id_is_rejected(
    order_id: str,
) -> None:
    with pytest.raises(
        RazorpayCheckoutInputError
    ):
        verify_checkout_signature(
            server_order_id=order_id,
            razorpay_payment_id=PAYMENT_ID,
            razorpay_signature=make_signature(),
            key_secret=KEY_SECRET,
        )


@pytest.mark.parametrize(
    "payment_id",
    [
        "",
        "not_a_payment",
        "pay_bad-value",
        "../pay_TestPayment12345",
    ],
)
def test_invalid_payment_id_is_rejected(
    payment_id: str,
) -> None:
    with pytest.raises(
        RazorpayCheckoutInputError
    ):
        verify_checkout_signature(
            server_order_id=ORDER_ID,
            razorpay_payment_id=payment_id,
            razorpay_signature=make_signature(),
            key_secret=KEY_SECRET,
        )


@pytest.mark.parametrize(
    "signature",
    [
        "",
        "not-a-signature",
        "a" * 63,
        "g" * 64,
    ],
)
def test_invalid_signature_format_is_rejected(
    signature: str,
) -> None:
    with pytest.raises(
        RazorpayCheckoutSignatureError
    ):
        verify_checkout_signature(
            server_order_id=ORDER_ID,
            razorpay_payment_id=PAYMENT_ID,
            razorpay_signature=signature,
            key_secret=KEY_SECRET,
        )


def test_missing_secret_is_rejected() -> None:
    with pytest.raises(
        RazorpayCheckoutInputError
    ):
        verify_checkout_signature(
            server_order_id=ORDER_ID,
            razorpay_payment_id=PAYMENT_ID,
            razorpay_signature=make_signature(),
            key_secret="",
        )


def test_request_rejects_forbidden_fields() -> None:
    with pytest.raises(ValidationError):
        RazorpayCheckoutVerificationRequest(
            razorpay_payment_id=PAYMENT_ID,
            razorpay_signature=make_signature(),
            customer_id="cust_forbidden",
        )


def test_result_exposes_no_secret_or_signature() -> None:
    result = verify_checkout_signature(
        server_order_id=ORDER_ID,
        razorpay_payment_id=PAYMENT_ID,
        razorpay_signature=make_signature(),
        key_secret=KEY_SECRET,
    )

    serialized = result.model_dump_json()

    assert KEY_SECRET not in serialized
    assert make_signature() not in serialized
    assert "key_secret" not in serialized
    assert "razorpay_signature" not in serialized