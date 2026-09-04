from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.evidence_contract import (
    CaseStrength,
    EvidenceCase,
    EvidenceContractError,
    EvidenceFact,
    EvidenceItem,
    EvidencePack,
    EvidenceStrength,
    FactSource,
    validate_pack_against_case,
)


UTC = timezone.utc


def make_case() -> EvidenceCase:
    return EvidenceCase(
        payment_id="pay_test_001",
        as_of=datetime(2026, 1, 20, tzinfo=UTC),
        deterministic_recommended_action="PREPARE_EVIDENCE",
        calibrated_probability=0.18,
        risk_band="CRITICAL",
        triggered_rule_codes=["SHIPMENT_SLA_BREACHED"],
        facts=[
            EvidenceFact(
                fact_id="PAYMENT_AMOUNT",
                source=FactSource.PAYMENT,
                label="Payment amount",
                value="INR 2,500.00",
            ),
            EvidenceFact(
                fact_id="SHIPMENT_DEADLINE",
                source=FactSource.SHIPMENT,
                label="Promised shipment deadline",
                value="2026-01-10T00:00:00+00:00",
                observed_at=datetime(2026, 1, 10, tzinfo=UTC),
            ),
        ],
    )


def make_pack() -> EvidencePack:
    return EvidencePack(
        payment_id="pay_test_001",
        case_summary=(
            "The payment has elevated chargeback risk and the promised "
            "shipment deadline has passed."
        ),
        case_strength=CaseStrength.MODERATE,
        recommended_action="PREPARE_EVIDENCE",
        action_rationale=(
            "Prepare timestamped fulfilment evidence for human review."
        ),
        evidence_items=[
            EvidenceItem(
                title="Shipment deadline passed",
                statement=(
                    "The promised shipment deadline was 10 January 2026."
                ),
                citation_fact_ids=["SHIPMENT_DEADLINE"],
                strength=EvidenceStrength.STRONG,
                relevance=(
                    "This supports early evidence preparation before a "
                    "possible non-delivery dispute."
                ),
            )
        ],
        limitations=[
            "This assessment uses synthetic data and requires review."
        ],
    )


def test_valid_pack_matches_case() -> None:
    validate_pack_against_case(make_pack(), make_case())


def test_pack_cannot_change_deterministic_action() -> None:
    case = make_case()
    pack = make_pack().model_copy(
        update={"recommended_action": "RECOMMEND_REFUND"}
    )

    with pytest.raises(EvidenceContractError, match="deterministic"):
        validate_pack_against_case(pack, case)


def test_unknown_citation_is_rejected() -> None:
    case = make_case()
    pack = make_pack()
    invalid_item = pack.evidence_items[0].model_copy(
        update={"citation_fact_ids": ["INVENTED_FACT"]}
    )
    pack = pack.model_copy(update={"evidence_items": [invalid_item]})

    with pytest.raises(EvidenceContractError, match="INVENTED_FACT"):
        validate_pack_against_case(pack, case)


def test_evidence_item_requires_a_citation() -> None:
    with pytest.raises(ValidationError):
        EvidenceItem(
            title="Missing citation",
            statement="This statement has no supporting source fact.",
            citation_fact_ids=[],
            strength=EvidenceStrength.WEAK,
            relevance="It must not be accepted without a citation.",
        )


def test_action_can_never_be_marked_executed() -> None:
    payload = make_pack().model_dump()
    payload["action_executed"] = True

    with pytest.raises(ValidationError):
        EvidencePack.model_validate(payload)


def test_unknown_output_fields_are_rejected() -> None:
    payload = make_pack().model_dump()
    payload["unverified_claim"] = "invented"

    with pytest.raises(ValidationError):
        EvidencePack.model_validate(payload)


def test_duplicate_input_fact_ids_are_rejected() -> None:
    duplicate = EvidenceFact(
        fact_id="PAYMENT_AMOUNT",
        source=FactSource.PAYMENT,
        label="Duplicate amount",
        value="INR 2,500.00",
    )
    payload = make_case().model_dump()
    payload["facts"].append(duplicate.model_dump())

    with pytest.raises(ValidationError, match="unique"):
        EvidenceCase.model_validate(payload)
