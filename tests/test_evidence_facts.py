from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.evidence_facts import (
    FORBIDDEN_INPUT_FIELDS,
    EvidenceFactError,
    build_evidence_case,
)


UTC = timezone.utc
AS_OF = datetime(2026, 1, 20, tzinfo=UTC)


def payment() -> dict:
    return {
        "payment_id": "pay_test_001",
        "customer_id": "cust_secret_001",
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "amount_paise": 250_000,
        "card_network": "Visa",
        "product_category": "electronics",
        "is_digital_good": False,
        "descriptor_clarity_score": 0.80,
        "phone_verified": True,
        "email_verified": True,
        "account_age_days": 365,
        "total_prior_orders": 5,
        "prior_disputes_count": 0,
        "txns_last_1h": 0,
        "txns_last_24h": 1,
        "txns_last_7d": 3,
        "amount_last_24h_paise": 250_000,
        "device_is_new": False,
        "ip_country_matches_billing": True,
        "ip_is_proxy_or_vpn": False,
        "cvv_result": "match",
        "threeds_status": "authenticated",
        "threeds_liability_shift": True,
        "billing_shipping_distance_km": 10,
        "is_duplicate_payment": False,
        "cancelled_subscription_billed": False,
        "calibrated_probability": 0.18,
        "recommended_action": "PREPARE_EVIDENCE",
        # Held-out outcome columns must be ignored.
        "chargeback_within_120d": 1,
        "chargeback_family": "true_fraud",
        "reason_code": "10.4",
    }


def operation() -> dict:
    return {
        "payment_id": "pay_test_001",
        "shipment_expected": True,
        "shipment_promised_at": datetime(2026, 1, 10, tzinfo=UTC),
        "shipment_delivered_at": None,
        "refund_requested_at": None,
        "refund_promised_by": None,
        "refund_processed_at": None,
        "shipment_sla_breached": True,
        "promised_refund_not_processed": True,
    }


def fact_ids(case) -> set[str]:
    return {fact.fact_id for fact in case.facts}


def test_builds_allowlisted_timestamped_case() -> None:
    case = build_evidence_case(
        payment(),
        operation=operation(),
        as_of=AS_OF,
    )

    assert case.payment_id == "pay_test_001"
    assert case.calibrated_probability == 0.18
    assert case.risk_band == "CRITICAL"
    assert "PAYMENT_AMOUNT" in fact_ids(case)
    assert "MODEL_CALIBRATED_PROBABILITY" in fact_ids(case)

def test_forbidden_fields_and_customer_id_never_reach_model() -> None:
    case = build_evidence_case(
        payment(),
        operation=operation(),
        as_of=AS_OF,
    )

    serialized = case.model_dump_json()

    assert "cust_secret_001" not in serialized
    assert "true_fraud" not in serialized
    assert "10.4" not in serialized

    forbidden_fact_ids = {
        field.upper()
        for field in FORBIDDEN_INPUT_FIELDS
    }

    assert not forbidden_fact_ids.intersection(
        fact_ids(case)
    )


def test_future_delivery_is_not_visible() -> None:
    data = operation()
    data["shipment_delivered_at"] = datetime(
        2026,
        1,
        25,
        tzinfo=UTC,
    )

    case = build_evidence_case(
        payment(),
        operation=data,
        as_of=AS_OF,
    )

    assert "SHIPMENT_DELIVERED_AT" not in fact_ids(case)
    assert "SHIPMENT_SLA_BREACHED" in case.triggered_rule_codes


def test_future_refund_processing_is_not_visible() -> None:
    data = operation()
    data.update(
        {
            "shipment_expected": False,
            "shipment_promised_at": None,
            "refund_requested_at": datetime(2026, 1, 5, tzinfo=UTC),
            "refund_promised_by": datetime(2026, 1, 10, tzinfo=UTC),
            "refund_processed_at": datetime(2026, 1, 25, tzinfo=UTC),
        }
    )

    case = build_evidence_case(
        payment(),
        operation=data,
        as_of=AS_OF,
    )

    assert "REFUND_PROCESSED_AT" not in fact_ids(case)
    assert "REFUND_NOT_PROCESSED" in case.triggered_rule_codes


def test_refund_breach_is_a_hard_deterministic_override() -> None:
    data = operation()
    data.update(
        {
            "shipment_expected": False,
            "shipment_promised_at": None,
            "refund_requested_at": datetime(2026, 1, 5, tzinfo=UTC),
            "refund_promised_by": datetime(2026, 1, 10, tzinfo=UTC),
            "refund_processed_at": None,
        }
    )

    case = build_evidence_case(
        payment(),
        operation=data,
        as_of=AS_OF,
    )

    assert case.deterministic_recommended_action == "RECOMMEND_REFUND"
    assert "LIFECYCLE_REFUND_NOT_PROCESSED" in fact_ids(case)


def test_capture_time_hard_override_is_preserved() -> None:
    data = payment()
    data["is_duplicate_payment"] = True

    case = build_evidence_case(
        data,
        operation=operation(),
        as_of=AS_OF,
    )

    assert case.deterministic_recommended_action == "RECOMMEND_REFUND"
    assert "CAPTURE_MERCHANT_DUPLICATE_PAYMENT" in fact_ids(case)


def test_mismatched_operation_is_rejected() -> None:
    data = operation()
    data["payment_id"] = "pay_someone_else"

    with pytest.raises(EvidenceFactError, match="does not belong"):
        build_evidence_case(
            payment(),
            operation=data,
            as_of=AS_OF,
        )


def test_as_of_cannot_precede_payment() -> None:
    with pytest.raises(EvidenceFactError, match="before the payment"):
        build_evidence_case(
            payment(),
            operation=operation(),
            as_of=datetime(2025, 12, 31, tzinfo=UTC),
        )


def test_invalid_policy_action_is_rejected() -> None:
    data = payment()
    data["recommended_action"] = "AUTO_REFUND"

    with pytest.raises(EvidenceFactError, match="recommended_action"):
        build_evidence_case(
            data,
            operation=operation(),
            as_of=AS_OF,
        )
