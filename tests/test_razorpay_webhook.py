from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from pydantic import SecretStr

from src.razorpay_webhook import (
    RazorpayWebhookPayloadError,
    RazorpayWebhookSignatureError,
    parse_verified_webhook,
)


SECRET = "test_webhook_secret_value"
EVENT_ID = "event-test-000001"


def raw_payment_event(
    *,
    event_type: str = "payment.captured",
) -> bytes:
    payload = {
        "entity": "event",
        "event": event_type,
        "created_at": 1788550000,
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_TestPayment12345",
                    "order_id": "order_TestOrder12345",
                    "amount": 10000,
                    "currency": "INR",
                    "status": "captured",
                    "method": "card",
                    "email": "forbidden@example.com",
                    "contact": "9999999999",
                }
            }
        },
    }

    return json.dumps(
        payload,
        separators=(",", ":"),
    ).encode("utf-8")


def signature_for(
    raw_body: bytes,
) -> str:
    return hmac.new(
        SECRET.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()


def test_valid_payment_webhook() -> None:
    raw_body = raw_payment_event()

    result = parse_verified_webhook(
        raw_body=raw_body,
        signature=signature_for(raw_body),
        event_id=EVENT_ID,
        webhook_secret=SECRET,
    )

    assert result.verified is True

    assert (
        result.event_type
        == "payment.captured"
    )

    assert (
        result.processing_decision
        == "PROCESS"
    )

    assert (
        result.payment_id
        == "pay_TestPayment12345"
    )

    assert (
        result.order_id
        == "order_TestOrder12345"
    )

    assert (
        result.financial_action_executed
        is False
    )


def test_secret_str_is_supported() -> None:
    raw_body = raw_payment_event()

    result = parse_verified_webhook(
        raw_body=raw_body,
        signature=signature_for(raw_body),
        event_id=EVENT_ID,
        webhook_secret=SecretStr(SECRET),
    )

    assert result.verified is True


def test_invalid_signature_is_rejected() -> None:
    raw_body = raw_payment_event()

    with pytest.raises(
        RazorpayWebhookSignatureError
    ):
        parse_verified_webhook(
            raw_body=raw_body,
            signature="a" * 64,
            event_id=EVENT_ID,
            webhook_secret=SECRET,
        )


def test_exact_raw_bytes_are_required() -> None:
    compact_body = raw_payment_event()

    pretty_body = json.dumps(
        json.loads(compact_body),
        indent=2,
    ).encode("utf-8")

    with pytest.raises(
        RazorpayWebhookSignatureError
    ):
        parse_verified_webhook(
            raw_body=pretty_body,
            signature=signature_for(
                compact_body
            ),
            event_id=EVENT_ID,
            webhook_secret=SECRET,
        )


def test_signature_checked_before_json() -> None:
    invalid_json = b"{not-json"

    with pytest.raises(
        RazorpayWebhookSignatureError
    ):
        parse_verified_webhook(
            raw_body=invalid_json,
            signature="a" * 64,
            event_id=EVENT_ID,
            webhook_secret=SECRET,
        )


def test_valid_signature_invalid_json_fails() -> None:
    invalid_json = b"{not-json"

    with pytest.raises(
        RazorpayWebhookPayloadError
    ):
        parse_verified_webhook(
            raw_body=invalid_json,
            signature=signature_for(
                invalid_json
            ),
            event_id=EVENT_ID,
            webhook_secret=SECRET,
        )


def test_invalid_event_id_is_rejected() -> None:
    raw_body = raw_payment_event()

    with pytest.raises(
        RazorpayWebhookPayloadError
    ):
        parse_verified_webhook(
            raw_body=raw_body,
            signature=signature_for(raw_body),
            event_id="../invalid",
            webhook_secret=SECRET,
        )


def test_unknown_event_is_safely_ignored() -> None:
    raw_body = raw_payment_event(
        event_type="payment.unknown_event"
    )

    result = parse_verified_webhook(
        raw_body=raw_body,
        signature=signature_for(raw_body),
        event_id=EVENT_ID,
        webhook_secret=SECRET,
    )

    assert (
        result.processing_decision
        == "IGNORE"
    )

    assert result.resource_kind == "none"
    assert result.resource_id is None


def test_pii_is_not_returned() -> None:
    raw_body = raw_payment_event()

    result = parse_verified_webhook(
        raw_body=raw_body,
        signature=signature_for(raw_body),
        event_id=EVENT_ID,
        webhook_secret=SECRET,
    )

    serialized = (
        result.model_dump_json()
    )

    assert SECRET not in serialized
    assert "forbidden@example.com" not in serialized
    assert "9999999999" not in serialized
    assert "razorpay_signature" not in serialized