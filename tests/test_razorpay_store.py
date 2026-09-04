import sqlite3

import pytest

from src.razorpay_store import (
    RazorpayIdempotencyConflictError,
    RazorpayStore,
    payload_sha256,
    raw_body_sha256,
)


def _order_request() -> dict:
    return {
        "amount": 250000,
        "currency": "INR",
        "receipt": "radar_demo_001",
        "notes": {
            "source": "chargeback_radar",
        },
    }


def _order_response() -> dict:
    return {
        "id": "order_Test123",
        "entity": "order",
        "amount": 250000,
        "currency": "INR",
        "status": "created",
    }


def _score_input() -> dict:
    return {
        "created_at": "2026-09-04T12:00:00+00:00",
        "amount_paise": 250000,
        "card_network": "Visa",
    }


def _score_response() -> dict:
    return {
        "calibrated_probability": 0.03,
        "recommended_action": "MONITOR",
        "action_executed": False,
    }


def test_new_order_reservation_is_atomic(
    tmp_path,
) -> None:
    store = RazorpayStore(
        tmp_path / "razorpay.sqlite3"
    )

    reservation = store.reserve_order(
        "request_0001",
        _order_request(),
    )

    assert reservation.state == "RESERVED"


def test_duplicate_pending_order_is_not_reserved_again(
    tmp_path,
) -> None:
    store = RazorpayStore(
        tmp_path / "razorpay.sqlite3"
    )

    first = store.reserve_order(
        "request_0001",
        _order_request(),
    )

    second = store.reserve_order(
        "request_0001",
        _order_request(),
    )

    assert first.state == "RESERVED"
    assert second.state == "PENDING"


def test_completed_order_is_replayed(
    tmp_path,
) -> None:
    store = RazorpayStore(
        tmp_path / "razorpay.sqlite3"
    )

    store.reserve_order(
        "request_0001",
        _order_request(),
    )

    store.complete_order(
        "request_0001",
        _order_request(),
        order_id="order_Test123",
        response_payload=_order_response(),
    )

    replay = store.reserve_order(
        "request_0001",
        _order_request(),
    )

    assert replay.state == "COMPLETED"
    assert replay.order_id == "order_Test123"
    assert replay.response == _order_response()


def test_order_key_cannot_be_reused_for_other_request(
    tmp_path,
) -> None:
    store = RazorpayStore(
        tmp_path / "razorpay.sqlite3"
    )

    store.reserve_order(
        "request_0001",
        _order_request(),
    )

    changed = _order_request()
    changed["amount"] = 500000

    with pytest.raises(
        RazorpayIdempotencyConflictError
    ):
        store.reserve_order(
            "request_0001",
            changed,
        )


def test_failed_order_uses_only_safe_error_code(
    tmp_path,
) -> None:
    database = tmp_path / "razorpay.sqlite3"

    store = RazorpayStore(database)

    store.reserve_order(
        "request_0001",
        _order_request(),
    )

    store.fail_order(
        "request_0001",
        _order_request(),
        error_code="RAZORPAY_UNAVAILABLE",
    )

    replay = store.reserve_order(
        "request_0001",
        _order_request(),
    )

    assert replay.state == "FAILED"
    assert (
        replay.error_code
        == "RAZORPAY_UNAVAILABLE"
    )

    with sqlite3.connect(database) as connection:
        stored = connection.execute(
            """
            SELECT error_code
            FROM order_requests
            """
        ).fetchone()[0]

    assert stored == "RAZORPAY_UNAVAILABLE"


def test_payment_score_round_trip(
    tmp_path,
) -> None:
    store = RazorpayStore(
        tmp_path / "razorpay.sqlite3"
    )

    assert (
        store.load_payment_score(
            "pay_Test123",
            _score_input(),
        )
        is None
    )

    stored = store.save_payment_score(
        "pay_Test123",
        _score_input(),
        _score_response(),
    )

    loaded = store.load_payment_score(
        "pay_Test123",
        _score_input(),
    )

    assert stored == _score_response()
    assert loaded == _score_response()


def test_duplicate_payment_save_is_idempotent(
    tmp_path,
) -> None:
    store = RazorpayStore(
        tmp_path / "razorpay.sqlite3"
    )

    first = store.save_payment_score(
        "pay_Test123",
        _score_input(),
        _score_response(),
    )

    second = store.save_payment_score(
        "pay_Test123",
        _score_input(),
        {
            "calibrated_probability": 0.99,
            "recommended_action": (
                "RECOMMEND_REFUND"
            ),
        },
    )

    assert first == second
    assert second == _score_response()


def test_payment_input_cannot_change(
    tmp_path,
) -> None:
    store = RazorpayStore(
        tmp_path / "razorpay.sqlite3"
    )

    store.save_payment_score(
        "pay_Test123",
        _score_input(),
        _score_response(),
    )

    changed = _score_input()
    changed["amount_paise"] = 999999

    with pytest.raises(
        RazorpayIdempotencyConflictError
    ):
        store.load_payment_score(
            "pay_Test123",
            changed,
        )


def test_first_webhook_is_reserved(
    tmp_path,
) -> None:
    store = RazorpayStore(
        tmp_path / "razorpay.sqlite3"
    )

    raw_body = b'{"event":"payment.captured"}'
    body_hash = raw_body_sha256(raw_body)

    first = store.reserve_webhook_event(
        event_id="event_0001",
        payload_hash=body_hash,
        event_type="payment.captured",
    )

    second = store.reserve_webhook_event(
        event_id="event_0001",
        payload_hash=body_hash,
        event_type="payment.captured",
    )

    assert first is True
    assert second is False


def test_webhook_event_id_cannot_change_payload(
    tmp_path,
) -> None:
    store = RazorpayStore(
        tmp_path / "razorpay.sqlite3"
    )

    store.reserve_webhook_event(
        event_id="event_0001",
        payload_hash=raw_body_sha256(b"first"),
        event_type="payment.captured",
    )

    with pytest.raises(
        RazorpayIdempotencyConflictError
    ):
        store.reserve_webhook_event(
            event_id="event_0001",
            payload_hash=(
                raw_body_sha256(b"second")
            ),
            event_type="payment.captured",
        )


def test_webhook_completion_is_recorded(
    tmp_path,
) -> None:
    store = RazorpayStore(
        tmp_path / "razorpay.sqlite3"
    )

    store.reserve_webhook_event(
        event_id="event_0001",
        payload_hash=raw_body_sha256(b"body"),
        event_type="payment.captured",
    )

    assert (
        store.webhook_event_status(
            "event_0001"
        )
        == "RECEIVED"
    )

    store.complete_webhook_event(
        "event_0001"
    )

    assert (
        store.webhook_event_status(
            "event_0001"
        )
        == "PROCESSED"
    )


def test_store_survives_new_instance(
    tmp_path,
) -> None:
    database = tmp_path / "razorpay.sqlite3"

    first = RazorpayStore(database)

    first.save_payment_score(
        "pay_Test123",
        _score_input(),
        _score_response(),
    )

    second = RazorpayStore(database)

    assert second.load_payment_score(
        "pay_Test123",
        _score_input(),
    ) == _score_response()


def test_hashing_is_deterministic() -> None:
    first = payload_sha256(
        {
            "amount": 100,
            "currency": "INR",
        }
    )

    second = payload_sha256(
        {
            "currency": "INR",
            "amount": 100,
        }
    )

    assert first == second
    assert len(first) == 64


def test_invalid_identifiers_are_rejected(
    tmp_path,
) -> None:
    store = RazorpayStore(
        tmp_path / "razorpay.sqlite3"
    )

    with pytest.raises(ValueError):
        store.reserve_order(
            "../unsafe",
            _order_request(),
        )

    with pytest.raises(ValueError):
        store.load_payment_score(
            "../../payment",
            _score_input(),
        )