from __future__ import annotations

from collections.abc import Iterable

from src.evidence_contract import (
    CaseStrength,
    EvidenceCase,
    EvidenceFact,
    EvidenceItem,
    EvidencePack,
    EvidenceStrength,
    FactSource,
    MissingEvidenceItem,
    MissingEvidencePriority,
    validate_pack_against_case,
)
from src.evidence_guardrails import (
    validate_evidence_case_safety,
    validate_generated_evidence_pack,
)


MAX_FALLBACK_EVIDENCE_ITEMS = 6

RULE_SOURCES = {
    FactSource.CAPTURE_RULE,
    FactSource.LIFECYCLE_RULE,
}

OPERATIONAL_SOURCES = {
    FactSource.PAYMENT,
    FactSource.CAPTURE_RULE,
    FactSource.LIFECYCLE_RULE,
    FactSource.ORDER,
    FactSource.SHIPMENT,
    FactSource.REFUND,
}

SOURCE_PRIORITY = {
    FactSource.CAPTURE_RULE: 0,
    FactSource.LIFECYCLE_RULE: 0,
    FactSource.REFUND: 1,
    FactSource.SHIPMENT: 1,
    FactSource.ORDER: 2,
    FactSource.PAYMENT: 3,
    FactSource.MODEL: 4,
    FactSource.POLICY: 5,
}

ACTION_RATIONALES = {
    "MONITOR": (
        "Continue monitoring under the deterministic policy; no automatic "
        "action has been taken."
    ),
    "PREPARE_EVIDENCE": (
        "Prepare the cited records for human review without contacting the "
        "customer automatically."
    ),
    "MANUAL_REVIEW": (
        "Route the cited records to a human analyst before any protected "
        "action is considered."
    ),
    "RECOMMEND_REFUND": (
        "Present the cited records and refund recommendation to a human "
        "approver; no refund has been executed."
    ),
}


def _ordered_facts(facts: Iterable[EvidenceFact]) -> list[EvidenceFact]:
    """Return a stable, operational-evidence-first ordering."""
    return sorted(
        facts,
        key=lambda fact: (
            SOURCE_PRIORITY[fact.source],
            fact.fact_id,
        ),
    )


def _evidence_strength(fact: EvidenceFact) -> EvidenceStrength:
    if fact.source in RULE_SOURCES:
        return EvidenceStrength.STRONG
    if fact.source in OPERATIONAL_SOURCES:
        return EvidenceStrength.MODERATE
    return EvidenceStrength.WEAK


def _relevance(fact: EvidenceFact) -> str:
    if fact.source in RULE_SOURCES:
        return (
            "This deterministic rule is direct operational support for "
            "human review."
        )
    if fact.source in {
        FactSource.ORDER,
        FactSource.SHIPMENT,
        FactSource.REFUND,
    }:
        return (
            "This timestamped operational record is relevant to the case "
            "timeline."
        )
    if fact.source == FactSource.PAYMENT:
        return (
            "This capture-time payment record provides transaction context "
            "for human review."
        )
    if fact.source == FactSource.MODEL:
        return (
            "This model estimate helps prioritize review but does not prove "
            "an outcome."
        )
    return (
        "This policy output records the deterministic recommendation but is "
        "not proof of an outcome."
    )


def _evidence_item(fact: EvidenceFact) -> EvidenceItem:
    title = fact.label if len(fact.label) >= 3 else f"Case fact {fact.label}"
    return EvidenceItem(
        title=title,
        statement=f"Recorded {fact.label}: {fact.value}.",
        citation_fact_ids=[fact.fact_id],
        strength=_evidence_strength(fact),
        relevance=_relevance(fact),
    )


def _case_strength(facts: list[EvidenceFact]) -> CaseStrength:
    rule_count = sum(fact.source in RULE_SOURCES for fact in facts)
    operational_count = sum(
        fact.source in OPERATIONAL_SOURCES
        for fact in facts
    )

    if rule_count:
        return CaseStrength.STRONG
    if operational_count >= 2:
        return CaseStrength.MODERATE
    if operational_count == 1:
        return CaseStrength.WEAK
    return CaseStrength.INSUFFICIENT


def _missing_evidence(fact_ids: set[str]) -> list[MissingEvidenceItem]:
    missing: list[MissingEvidenceItem] = []

    shipment_is_relevant = bool(
        {
            "SHIPMENT_EXPECTED",
            "SHIPMENT_PROMISED_AT",
            "LIFECYCLE_SHIPMENT_SLA_BREACHED",
        }
        & fact_ids
    )
    if shipment_is_relevant and "SHIPMENT_DELIVERED_AT" not in fact_ids:
        missing.append(
            MissingEvidenceItem(
                item="Carrier delivery confirmation",
                reason=(
                    "The available facts do not contain a completed delivery "
                    "record for the shipment timeline."
                ),
                expected_source="Carrier or shipment provider",
                priority=MissingEvidencePriority.RECOMMENDED,
            )
        )

    refund_is_relevant = bool(
        {
            "REFUND_REQUESTED_AT",
            "REFUND_PROMISED_BY",
            "LIFECYCLE_REFUND_NOT_PROCESSED",
        }
        & fact_ids
    )
    if refund_is_relevant and "REFUND_PROCESSED_AT" not in fact_ids:
        missing.append(
            MissingEvidenceItem(
                item="Refund processing confirmation",
                reason=(
                    "The available facts do not contain confirmation that the "
                    "requested refund was processed."
                ),
                expected_source="Payment gateway or refund ledger",
                priority=MissingEvidencePriority.REQUIRED,
            )
        )

    return missing


def generate_fallback_evidence_pack(
    evidence_case: EvidenceCase,
) -> EvidencePack:
    """Build a deterministic, grounded draft when live AI is unavailable."""
    validate_evidence_case_safety(evidence_case)

    ordered_facts = _ordered_facts(evidence_case.facts)
    selected_facts = ordered_facts[:MAX_FALLBACK_EVIDENCE_ITEMS]
    fact_ids = {fact.fact_id for fact in evidence_case.facts}

    pack = EvidencePack(
        payment_id=evidence_case.payment_id,
        case_summary=(
            "The available timestamped records support a human-reviewed "
            "evidence draft. The deterministic policy recommendation is "
            "preserved and no action has been executed."
        ),
        case_strength=_case_strength(evidence_case.facts),
        recommended_action=(
            evidence_case.deterministic_recommended_action
        ),
        action_rationale=ACTION_RATIONALES[
            evidence_case.deterministic_recommended_action
        ],
        evidence_items=[
            _evidence_item(fact)
            for fact in selected_facts
        ],
        missing_evidence=_missing_evidence(fact_ids),
        customer_message=None,
        limitations=[
            (
                "This deterministic fallback did not use the OpenAI model "
                "because the live drafting path was unavailable."
            ),
            (
                "The risk estimate comes from a model trained and evaluated "
                "on synthetic data; a human must verify applicability."
            ),
        ],
        human_approval_required=True,
        action_executed=False,
    )

    validate_pack_against_case(pack, evidence_case)
    validate_generated_evidence_pack(pack, evidence_case)
    return pack
