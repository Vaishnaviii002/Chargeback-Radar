from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from src.razorpay_store import (
    RazorpayIdempotencyConflictError,
    RazorpayStore,
    raw_body_sha256,
)
from src.razorpay_webhook import (
    RazorpayWebhookConfigurationError,
    RazorpayWebhookSignatureError,
)
from src.razorpay_webhook_service import (
    RazorpayWebhookConfig,
    RazorpayWebhookDisabledError,
    RazorpayWebhookService,
)


SECRET = "test_webhook_secret_value"
EVENT_ID = "event_test_000001"


def event_body(
    *,
    event_type: str = "payment.captured",
    created_at: int = 1788550000,
) -> bytes:
    document = {
        "entity": "event",
        "event": event_type,
        "created_at": created_at,
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_TestPayment12345",
                    "order_id": (
                        "order_TestOrder12345"
                    ),
                    "amount": 10000,
                    "currency": "INR",
                    "status": "captured",
                    "method": "card",
                    "email": (
                        "forbidden@example.com"
                    ),
                    "contact": "9999999999",
                }
            }
        },
    }

    return json.dumps(
        document,
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


def make_service(
    database_path: Path,
    *,
    enabled: bool = True,
    secret: str | None = SECRET,
) -> RazorpayWebhookService:
    return RazorpayWebhookService(
        store=RazorpayStore(
            database_path
        ),
        config=RazorpayWebhookConfig(
            enabled=enabled,
            webhook_secret=(
                SecretStr(secret)
                if secret is not None
                else None
            ),
        ),
    )


def test_new_event_is_processed_once(
    tmp_path: Path,
) -> None:
    database_path = (
        tmp_path / "razorpay.sqlite3"
    )

    service = make_service(
        database_path
    )

    raw_body = event_body()

    result = service.process(
        raw_body=raw_body,
        signature=signature_for(raw_body),
        event_id=EVENT_ID,
    )

    assert (
        result.delivery_mode
        == "PROCESSED"
    )

    assert result.duplicate is False

    assert (
        result.event.payment_id
        == "pay_TestPayment12345"
    )

    assert (
        service.store.webhook_event_status(
            EVENT_ID
        )
        == "PROCESSED"
    )

    assert (
        result.financial_action_executed
        is False
    )


def test_duplicate_event_is_replayed(
    tmp_path: Path,
) -> None:
    service = make_service(
        tmp_path / "razorpay.sqlite3"
    )

    raw_body = event_body()
    signature = signature_for(raw_body)

    first = service.process(
        raw_body=raw_body,
        signature=signature,
        event_id=EVENT_ID,
    )

    second = service.process(
        raw_body=raw_body,
        signature=signature,
        event_id=EVENT_ID,
    )

    assert (
        first.delivery_mode
        == "PROCESSED"
    )

    assert (
        second.delivery_mode
        == "IDEMPOTENT_REPLAY"
    )

    assert second.duplicate is True


def test_reused_event_id_with_changed_body_fails(
    tmp_path: Path,
) -> None:
    service = make_service(
        tmp_path / "razorpay.sqlite3"
    )

    original = event_body(
        event_type="payment.captured"
    )

    changed = event_body(
        event_type="payment.authorized",
        created_at=1788550001,
    )

    service.process(
        raw_body=original,
        signature=signature_for(original),
        event_id=EVENT_ID,
    )

    with pytest.raises(
        RazorpayIdempotencyConflictError
    ):
        service.process(
            raw_body=changed,
            signature=signature_for(changed),
            event_id=EVENT_ID,
        )


def test_invalid_signature_is_not_reserved(
    tmp_path: Path,
) -> None:
    service = make_service(
        tmp_path / "razorpay.sqlite3"
    )

    raw_body = event_body()

    with pytest.raises(
        RazorpayWebhookSignatureError
    ):
        service.process(
            raw_body=raw_body,
            signature="a" * 64,
            event_id=EVENT_ID,
        )

    assert (
        service.store.webhook_event_status(
            EVENT_ID
        )
        is None
    )


def test_unknown_event_is_recorded_as_ignored(
    tmp_path: Path,
) -> None:
    service = make_service(
        tmp_path / "razorpay.sqlite3"
    )

    raw_body = event_body(
        event_type="payment.unknown_event"
    )

    result = service.process(
        raw_body=raw_body,
        signature=signature_for(raw_body),
        event_id=EVENT_ID,
    )

    assert result.delivery_mode == "IGNORED"

    assert (
        result.event.processing_decision
        == "IGNORE"
    )

    assert (
        service.store.webhook_event_status(
            EVENT_ID
        )
        == "PROCESSED"
    )


def test_received_event_can_recover_safely(
    tmp_path: Path,
) -> None:
    service = make_service(
        tmp_path / "razorpay.sqlite3"
    )

    raw_body = event_body()

    reserved = (
        service.store.reserve_webhook_event(
            event_id=EVENT_ID,
            payload_hash=raw_body_sha256(
                raw_body
            ),
            event_type="payment.captured",
        )
    )

    assert reserved is True

    assert (
        service.store.webhook_event_status(
            EVENT_ID
        )
        == "RECEIVED"
    )

    result = service.process(
        raw_body=raw_body,
        signature=signature_for(raw_body),
        event_id=EVENT_ID,
    )

    assert (
        result.delivery_mode
        == "IDEMPOTENT_REPLAY"
    )

    assert (
        service.store.webhook_event_status(
            EVENT_ID
        )
        == "PROCESSED"
    )


def test_missing_secret_fails_closed(
    tmp_path: Path,
) -> None:
    service = make_service(
        tmp_path / "razorpay.sqlite3",
        secret=None,
    )

    raw_body = event_body()

    with pytest.raises(
        RazorpayWebhookConfigurationError
    ):
        service.process(
            raw_body=raw_body,
            signature=signature_for(raw_body),
            event_id=EVENT_ID,
        )


def test_disabled_service_fails_closed(
    tmp_path: Path,
) -> None:
    service = make_service(
        tmp_path / "razorpay.sqlite3",
        enabled=False,
    )

    raw_body = event_body()

    with pytest.raises(
        RazorpayWebhookDisabledError
    ):
        service.process(
            raw_body=raw_body,
            signature=signature_for(raw_body),
            event_id=EVENT_ID,
        )


def test_result_contains_no_secret_or_pii(
    tmp_path: Path,
) -> None:
    service = make_service(
        tmp_path / "razorpay.sqlite3"
    )

    raw_body = event_body()

    result = service.process(
        raw_body=raw_body,
        signature=signature_for(raw_body),
        event_id=EVENT_ID,
    )

    serialized = (
        result.model_dump_json()
    )

    assert SECRET not in serialized

    assert (
        "forbidden@example.com"
        not in serialized
    )

    assert "9999999999" not in serialized
    assert "razorpay_signature" not in serialized
    assert "key_secret" not in serialized