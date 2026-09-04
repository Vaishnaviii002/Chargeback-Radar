from __future__ import annotations

import hashlib
import hmac
import json
import re
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
)


MAX_WEBHOOK_BODY_BYTES = 1_000_000

EVENT_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9_-]{6,128}$"
)
RESOURCE_ID_PATTERN = re.compile(
    r"^(pay|order|disp)_[A-Za-z0-9_-]{3,96}$"
)

SIGNATURE_PATTERN = re.compile(
    r"^[A-Fa-f0-9]{64}$"
)

SUPPORTED_EVENTS = {
    "payment.authorized",
    "payment.captured",
    "payment.failed",
    "order.paid",
    "payment.dispute.created",
    "payment.dispute.won",
    "payment.dispute.lost",
    "payment.dispute.closed",
}


class RazorpayWebhookError(RuntimeError):
    pass


class RazorpayWebhookConfigurationError(
    RazorpayWebhookError
):
    pass


class RazorpayWebhookSignatureError(
    RazorpayWebhookError
):
    pass


class RazorpayWebhookPayloadError(
    RazorpayWebhookError
):
    pass


class RazorpayVerifiedWebhook(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    verified: Literal[True] = True

    event_id: str = Field(
        min_length=8,
        max_length=128,
    )

    event_type: str = Field(
        min_length=3,
        max_length=100,
    )

    created_at: int = Field(
        ge=0,
    )

    processing_decision: Literal[
        "PROCESS",
        "IGNORE",
    ]

    resource_kind: Literal[
        "payment",
        "order",
        "dispute",
        "none",
    ]

    resource_id: str | None = None
    payment_id: str | None = None
    order_id: str | None = None

    signature_algorithm: Literal[
        "HMAC_SHA256"
    ] = "HMAC_SHA256"

    source: Literal[
        "RAZORPAY_WEBHOOK"
    ] = "RAZORPAY_WEBHOOK"

    financial_action_executed: Literal[
        False
    ] = False


def _secret_value(
    webhook_secret: SecretStr | str,
) -> str:
    if isinstance(webhook_secret, SecretStr):
        secret = webhook_secret.get_secret_value()

    elif isinstance(webhook_secret, str):
        secret = webhook_secret

    else:
        raise RazorpayWebhookConfigurationError(
            "Razorpay webhook secret is unavailable."
        )

    if len(secret) < 8:
        raise RazorpayWebhookConfigurationError(
            "Razorpay webhook secret is unavailable."
        )

    return secret


def verify_webhook_signature(
    *,
    raw_body: bytes,
    signature: str,
    webhook_secret: SecretStr | str,
) -> None:
    if not isinstance(raw_body, bytes):
        raise RazorpayWebhookPayloadError(
            "Webhook body must be raw bytes."
        )

    if not raw_body:
        raise RazorpayWebhookPayloadError(
            "Webhook body is empty."
        )

    if (
        len(raw_body)
        > MAX_WEBHOOK_BODY_BYTES
    ):
        raise RazorpayWebhookPayloadError(
            "Webhook body exceeds the size limit."
        )

    if not SIGNATURE_PATTERN.fullmatch(
        signature
    ):
        raise RazorpayWebhookSignatureError(
            "Invalid Razorpay webhook signature."
        )

    secret = _secret_value(
        webhook_secret
    )

    expected_signature = hmac.new(
        secret.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(
        expected_signature,
        signature.lower(),
    ):
        raise RazorpayWebhookSignatureError(
            "Invalid Razorpay webhook signature."
        )


def _dictionary(
    value: Any,
    *,
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RazorpayWebhookPayloadError(
            f"{field_name} must be an object."
        )

    return value


def _resource_entity(
    payload: dict[str, Any],
    resource_kind: str,
) -> dict[str, Any]:
    wrapper = _dictionary(
        payload.get(resource_kind),
        field_name=(
            f"payload.{resource_kind}"
        ),
    )

    return _dictionary(
        wrapper.get("entity"),
        field_name=(
            f"payload.{resource_kind}.entity"
        ),
    )


def _safe_resource_id(
    value: Any,
    *,
    field_name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not RESOURCE_ID_PATTERN.fullmatch(
            value
        )
    ):
        raise RazorpayWebhookPayloadError(
            f"{field_name} is invalid."
        )

    return value


def parse_verified_webhook(
    *,
    raw_body: bytes,
    signature: str,
    event_id: str,
    webhook_secret: SecretStr | str,
) -> RazorpayVerifiedWebhook:
    # Authentication must happen before JSON parsing.
    verify_webhook_signature(
        raw_body=raw_body,
        signature=signature,
        webhook_secret=webhook_secret,
    )

    if not EVENT_ID_PATTERN.fullmatch(
        event_id
    ):
        raise RazorpayWebhookPayloadError(
            "X-Razorpay-Event-Id is invalid."
        )

    try:
        decoded = raw_body.decode("utf-8")
        document = json.loads(decoded)

    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise RazorpayWebhookPayloadError(
            "Webhook body is not valid UTF-8 JSON."
        ) from error

    document = _dictionary(
        document,
        field_name="webhook",
    )

    if document.get("entity") != "event":
        raise RazorpayWebhookPayloadError(
            "Webhook entity must be event."
        )

    event_type = document.get("event")

    if (
        not isinstance(event_type, str)
        or not event_type
        or len(event_type) > 100
    ):
        raise RazorpayWebhookPayloadError(
            "Webhook event type is invalid."
        )

    created_at = document.get("created_at")

    if (
        type(created_at) is not int
        or created_at < 0
    ):
        raise RazorpayWebhookPayloadError(
            "Webhook created_at is invalid."
        )

    if event_type not in SUPPORTED_EVENTS:
        return RazorpayVerifiedWebhook(
            event_id=event_id,
            event_type=event_type,
            created_at=created_at,
            processing_decision="IGNORE",
            resource_kind="none",
            resource_id=None,
            payment_id=None,
            order_id=None,
            financial_action_executed=False,
        )

    payload = _dictionary(
        document.get("payload"),
        field_name="payload",
    )

    if event_type.startswith(
        "payment.dispute."
    ):
        dispute = _resource_entity(
            payload,
            "dispute",
        )

        dispute_id = _safe_resource_id(
            dispute.get("id"),
            field_name="dispute.id",
        )

        payment_id = _safe_resource_id(
            dispute.get("payment_id"),
            field_name="dispute.payment_id",
        )

        return RazorpayVerifiedWebhook(
            event_id=event_id,
            event_type=event_type,
            created_at=created_at,
            processing_decision="PROCESS",
            resource_kind="dispute",
            resource_id=dispute_id,
            payment_id=payment_id,
            order_id=None,
            financial_action_executed=False,
        )

    if event_type.startswith("payment."):
        payment = _resource_entity(
            payload,
            "payment",
        )

        payment_id = _safe_resource_id(
            payment.get("id"),
            field_name="payment.id",
        )

        order_value = payment.get(
            "order_id"
        )

        order_id = (
            _safe_resource_id(
                order_value,
                field_name="payment.order_id",
            )
            if order_value is not None
            else None
        )

        return RazorpayVerifiedWebhook(
            event_id=event_id,
            event_type=event_type,
            created_at=created_at,
            processing_decision="PROCESS",
            resource_kind="payment",
            resource_id=payment_id,
            payment_id=payment_id,
            order_id=order_id,
            financial_action_executed=False,
        )

    order = _resource_entity(
        payload,
        "order",
    )

    order_id = _safe_resource_id(
        order.get("id"),
        field_name="order.id",
    )

    return RazorpayVerifiedWebhook(
        event_id=event_id,
        event_type=event_type,
        created_at=created_at,
        processing_decision="PROCESS",
        resource_kind="order",
        resource_id=order_id,
        payment_id=None,
        order_id=order_id,
        financial_action_executed=False,
    )