from __future__ import annotations

import hashlib
import hmac
import re
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
)


ORDER_ID_PATTERN = re.compile(
    r"^order_[A-Za-z0-9]{6,64}$"
)

PAYMENT_ID_PATTERN = re.compile(
    r"^pay_[A-Za-z0-9]{6,64}$"
)

SIGNATURE_PATTERN = re.compile(
    r"^[A-Fa-f0-9]{64}$"
)


class RazorpayCheckoutVerificationError(
    RuntimeError
):
    pass


class RazorpayCheckoutInputError(
    RazorpayCheckoutVerificationError
):
    pass


class RazorpayCheckoutSignatureError(
    RazorpayCheckoutVerificationError
):
    pass


class RazorpayCheckoutVerificationRequest(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    razorpay_payment_id: str = Field(
        min_length=10,
        max_length=80,
        pattern=r"^pay_[A-Za-z0-9]{6,64}$",
    )

    razorpay_signature: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[A-Fa-f0-9]{64}$",
        repr=False,
    )


class RazorpayCheckoutVerificationResult(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    verified: Literal[True] = True

    razorpay_order_id: str
    razorpay_payment_id: str

    algorithm: Literal[
        "HMAC_SHA256"
    ] = "HMAC_SHA256"

    source: Literal[
        "RAZORPAY_TEST_MODE"
    ] = "RAZORPAY_TEST_MODE"

    synthetic_evaluation_affected: Literal[
        False
    ] = False

    financial_action_executed: Literal[
        False
    ] = False


def _secret_value(
    key_secret: SecretStr | str,
) -> str:
    if isinstance(key_secret, SecretStr):
        secret = key_secret.get_secret_value()
    elif isinstance(key_secret, str):
        secret = key_secret
    else:
        raise RazorpayCheckoutInputError(
            "Razorpay key secret is unavailable."
        )

    if len(secret) < 8:
        raise RazorpayCheckoutInputError(
            "Razorpay key secret is unavailable."
        )

    return secret


def verify_checkout_signature(
    *,
    server_order_id: str,
    razorpay_payment_id: str,
    razorpay_signature: str,
    key_secret: SecretStr | str,
) -> RazorpayCheckoutVerificationResult:
    """
    Verify a Razorpay Standard Checkout success response.

    server_order_id must come from our trusted server-side order
    record or endpoint path. It must not be replaced with an
    order ID supplied by the browser callback.
    """

    if not ORDER_ID_PATTERN.fullmatch(
        server_order_id
    ):
        raise RazorpayCheckoutInputError(
            "Invalid server-owned Razorpay order ID."
        )

    if not PAYMENT_ID_PATTERN.fullmatch(
        razorpay_payment_id
    ):
        raise RazorpayCheckoutInputError(
            "Invalid Razorpay payment ID."
        )

    if not SIGNATURE_PATTERN.fullmatch(
        razorpay_signature
    ):
        raise RazorpayCheckoutSignatureError(
            "Invalid Razorpay Checkout signature."
        )

    secret = _secret_value(key_secret)

    signed_payload = (
        f"{server_order_id}|"
        f"{razorpay_payment_id}"
    ).encode("utf-8")

    expected_signature = hmac.new(
        secret.encode("utf-8"),
        signed_payload,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(
        expected_signature,
        razorpay_signature.lower(),
    ):
        raise RazorpayCheckoutSignatureError(
            "Invalid Razorpay Checkout signature."
        )

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