from __future__ import annotations

from datetime import datetime, timezone

from src.evidence_contract import (
    CaseStrength,
    EvidenceCase,
    EvidenceFact,
    EvidenceStrength,
    FactSource,
)
from src.evidence_fallback import generate_fallback_evidence_pack
from src.evidence_guardrails import validate_generated_evidence_pack


UTC = timezone.utc


def make_case(
    *,
    action: str = "PREPARE_EVIDENCE",
    facts: list[EvidenceFact] | None = None,
) -> EvidenceCase:
    return EvidenceCase(
        payment_id="pay_test_001",
        as_of=datetime(2026, 1, 20, tzinfo=UTC),
        deterministic_recommended_action=action,
        calibrated_probability=0.18,
        risk_band="CRITICAL",
        triggered_rule_codes=["SHIPMENT_SLA_BREACHED"],
        facts=facts
        or [
            EvidenceFact(
                fact_id="PAYMENT_AMOUNT",
                source=FactSource.PAYMENT,
                label="Payment amount",
                value="INR 2,500.00",
            ),
            EvidenceFact(
                fact_id="SHIPMENT_PROMISED_AT",
                source=FactSource.SHIPMENT,
                label="Promised shipment deadline",
                value="2026-01-10T00:00:00+00:00",
            ),
            EvidenceFact(
                fact_id="LIFECYCLE_SHIPMENT_SLA_BREACHED",
                source=FactSource.LIFECYCLE_RULE,
                label="Shipment SLA rule",
                value="Shipment deadline passed without delivery evidence",
            ),
            EvidenceFact(
                fact_id="MODEL_CALIBRATED_PROBABILITY",
                source=FactSource.MODEL,
                label="Calibrated chargeback probability",
                value="18.00%",
            ),
        ],
    )


def test_fallback_is_deterministic() -> None:
    case = make_case()

    first = generate_fallback_evidence_pack(case)
    second = generate_fallback_evidence_pack(case)

    assert first.model_dump() == second.model_dump()


def test_fallback_passes_all_output_guardrails() -> None:
    case = make_case()
    pack = generate_fallback_evidence_pack(case)

    report = validate_generated_evidence_pack(pack, case)

    assert report.passed is True
    assert report.violations == []


def test_fallback_preserves_every_policy_action() -> None:
    for action in (
        "MONITOR",
        "PREPARE_EVIDENCE",
        "MANUAL_REVIEW",
        "RECOMMEND_REFUND",
    ):
        case = make_case(action=action)
        pack = generate_fallback_evidence_pack(case)

        assert pack.recommended_action == action


def test_fallback_never_executes_or_sends() -> None:
    pack = generate_fallback_evidence_pack(make_case())

    assert pack.human_approval_required is True
    assert pack.action_executed is False
    assert pack.customer_message is None


def test_rule_fact_is_prioritized_and_strong() -> None:
    pack = generate_fallback_evidence_pack(make_case())

    first = pack.evidence_items[0]
    assert first.citation_fact_ids == [
        "LIFECYCLE_SHIPMENT_SLA_BREACHED"
    ]
    assert first.strength == EvidenceStrength.STRONG


def test_model_only_evidence_is_weak_and_case_is_insufficient() -> None:
    case = make_case(
        facts=[
            EvidenceFact(
                fact_id="MODEL_CALIBRATED_PROBABILITY",
                source=FactSource.MODEL,
                label="Calibrated chargeback probability",
                value="18.00%",
            )
        ]
    )

    pack = generate_fallback_evidence_pack(case)

    assert pack.case_strength == CaseStrength.INSUFFICIENT
    assert pack.evidence_items[0].strength == EvidenceStrength.WEAK


def test_missing_delivery_confirmation_is_disclosed() -> None:
    pack = generate_fallback_evidence_pack(make_case())

    assert any(
        item.item == "Carrier delivery confirmation"
        for item in pack.missing_evidence
    )


def test_missing_refund_confirmation_is_disclosed() -> None:
    case = make_case(
        action="RECOMMEND_REFUND",
        facts=[
            EvidenceFact(
                fact_id="REFUND_REQUESTED_AT",
                source=FactSource.REFUND,
                label="Refund requested at",
                value="2026-01-05T00:00:00+00:00",
            ),
            EvidenceFact(
                fact_id="REFUND_PROMISED_BY",
                source=FactSource.REFUND,
                label="Refund promised by",
                value="2026-01-10T00:00:00+00:00",
            ),
        ],
    )

    pack = generate_fallback_evidence_pack(case)

    assert any(
        item.item == "Refund processing confirmation"
        for item in pack.missing_evidence
    )


def test_fallback_discloses_its_provenance_and_model_limit() -> None:
    pack = generate_fallback_evidence_pack(make_case())
    limitations = " ".join(pack.limitations).lower()

    assert "deterministic fallback" in limitations
    assert "openai model" in limitations
    assert "synthetic data" in limitations
