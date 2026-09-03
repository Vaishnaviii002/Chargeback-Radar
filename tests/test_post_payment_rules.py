from __future__ import annotations

from datetime import datetime, timezone

from src.post_payment_rules import evaluate_post_payment_rules


UTC = timezone.utc


def rule_codes(result: dict) -> set[str]:
    return {hit["code"] for hit in result["hits"]}


def base_operation() -> dict:
    return {
        "payment_id": "pay_test_001",
        "shipment_expected": True,
        "shipment_promised_at": datetime(2026, 1, 10, tzinfo=UTC),
        "shipment_delivered_at": None,
        "refund_requested_at": None,
        "refund_promised_by": None,
        "refund_processed_at": None,
        # These precomputed future flags must never control the rule result.
        "shipment_sla_breached": True,
        "promised_refund_not_processed": True,
    }


def test_shipment_rule_does_not_fire_before_deadline() -> None:
    operation = base_operation()

    result = evaluate_post_payment_rules(
        operation,
        as_of=datetime(2026, 1, 9, tzinfo=UTC),
    )

    assert "SHIPMENT_SLA_BREACHED" not in rule_codes(result)


def test_missing_shipment_fires_after_deadline() -> None:
    operation = base_operation()

    result = evaluate_post_payment_rules(
        operation,
        as_of=datetime(2026, 1, 11, tzinfo=UTC),
    )

    assert "SHIPMENT_SLA_BREACHED" in rule_codes(result)
    assert result["hard_override_action"] is None


def test_on_time_shipment_does_not_fire() -> None:
    operation = base_operation()
    operation["shipment_delivered_at"] = datetime(
        2026,
        1,
        9,
        tzinfo=UTC,
    )

    result = evaluate_post_payment_rules(
        operation,
        as_of=datetime(2026, 1, 12, tzinfo=UTC),
    )

    assert "SHIPMENT_SLA_BREACHED" not in rule_codes(result)


def test_late_shipment_fires_after_deadline() -> None:
    operation = base_operation()
    operation["shipment_delivered_at"] = datetime(
        2026,
        1,
        12,
        tzinfo=UTC,
    )

    result = evaluate_post_payment_rules(
        operation,
        as_of=datetime(2026, 1, 13, tzinfo=UTC),
    )

    assert "SHIPMENT_SLA_BREACHED" in rule_codes(result)


def test_refund_rule_does_not_fire_before_deadline() -> None:
    operation = base_operation()
    operation.update(
        {
            "shipment_expected": False,
            "refund_requested_at": datetime(2026, 1, 5, tzinfo=UTC),
            "refund_promised_by": datetime(2026, 1, 10, tzinfo=UTC),
        }
    )

    result = evaluate_post_payment_rules(
        operation,
        as_of=datetime(2026, 1, 9, tzinfo=UTC),
    )

    assert "REFUND_NOT_PROCESSED" not in rule_codes(result)


def test_missing_refund_fires_after_deadline() -> None:
    operation = base_operation()
    operation.update(
        {
            "shipment_expected": False,
            "refund_requested_at": datetime(2026, 1, 5, tzinfo=UTC),
            "refund_promised_by": datetime(2026, 1, 10, tzinfo=UTC),
        }
    )

    result = evaluate_post_payment_rules(
        operation,
        as_of=datetime(2026, 1, 11, tzinfo=UTC),
    )

    assert "REFUND_NOT_PROCESSED" in rule_codes(result)
    assert result["hard_override_action"] == "RECOMMEND_REFUND"
    assert result["requires_human_approval"] is True
    assert result["action_executed"] is False


def test_on_time_refund_does_not_fire() -> None:
    operation = base_operation()
    operation.update(
        {
            "shipment_expected": False,
            "refund_requested_at": datetime(2026, 1, 5, tzinfo=UTC),
            "refund_promised_by": datetime(2026, 1, 10, tzinfo=UTC),
            "refund_processed_at": datetime(2026, 1, 9, tzinfo=UTC),
        }
    )

    result = evaluate_post_payment_rules(
        operation,
        as_of=datetime(2026, 1, 11, tzinfo=UTC),
    )

    assert "REFUND_NOT_PROCESSED" not in rule_codes(result)


def test_future_refund_timestamp_is_not_used_early() -> None:
    operation = base_operation()
    operation.update(
        {
            "shipment_expected": False,
            "refund_requested_at": datetime(2026, 1, 5, tzinfo=UTC),
            "refund_promised_by": datetime(2026, 1, 10, tzinfo=UTC),
            "refund_processed_at": datetime(2026, 1, 15, tzinfo=UTC),
        }
    )

    result = evaluate_post_payment_rules(
        operation,
        as_of=datetime(2026, 1, 11, tzinfo=UTC),
    )

    assert "REFUND_NOT_PROCESSED" in rule_codes(result)
    evidence = result["hits"][0]["evidence"]
    assert evidence["refund_processed_at"] is None


def test_precomputed_future_flags_are_ignored() -> None:
    operation = base_operation()
    operation.update(
        {
            "shipment_expected": False,
            "shipment_promised_at": None,
        }
    )

    result = evaluate_post_payment_rules(
        operation,
        as_of=datetime(2026, 1, 20, tzinfo=UTC),
    )

    assert result["triggered"] is False
    assert result["hits"] == []
