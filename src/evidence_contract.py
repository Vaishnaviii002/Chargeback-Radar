from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class FactSource(str, Enum):
    PAYMENT = "PAYMENT"
    MODEL = "MODEL"
    POLICY = "POLICY"
    CAPTURE_RULE = "CAPTURE_RULE"
    LIFECYCLE_RULE = "LIFECYCLE_RULE"
    ORDER = "ORDER"
    SHIPMENT = "SHIPMENT"
    REFUND = "REFUND"


class EvidenceStrength(str, Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"


class CaseStrength(str, Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    INSUFFICIENT = "INSUFFICIENT"


class MissingEvidencePriority(str, Enum):
    REQUIRED = "REQUIRED"
    RECOMMENDED = "RECOMMENDED"
    OPTIONAL = "OPTIONAL"


class EvidenceFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact_id: str = Field(
        pattern=r"^[A-Z][A-Z0-9_]{2,63}$"
    )
    source: FactSource
    label: str = Field(min_length=2, max_length=120)
    value: str = Field(min_length=1, max_length=500)
    observed_at: datetime | None = None


class EvidenceCase(BaseModel):
    """Ground-truth envelope supplied to the drafting model."""

    model_config = ConfigDict(extra="forbid")

    payment_id: str = Field(min_length=4, max_length=100)
    as_of: datetime
    deterministic_recommended_action: Literal[
        "MONITOR",
        "PREPARE_EVIDENCE",
        "MANUAL_REVIEW",
        "RECOMMEND_REFUND",
    ]
    calibrated_probability: float = Field(ge=0, le=1)
    risk_band: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    triggered_rule_codes: list[str] = Field(
        default_factory=list,
        max_length=20,
    )
    facts: list[EvidenceFact] = Field(
        min_length=1,
        max_length=60,
    )

    @model_validator(mode="after")
    def fact_ids_must_be_unique(self) -> "EvidenceCase":
        fact_ids = [fact.fact_id for fact in self.facts]

        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("Evidence fact IDs must be unique.")

        return self


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=3, max_length=120)
    statement: str = Field(min_length=10, max_length=700)
    citation_fact_ids: list[str] = Field(
        min_length=1,
        max_length=8,
    )
    strength: EvidenceStrength
    relevance: str = Field(min_length=10, max_length=500)


class MissingEvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item: str = Field(min_length=3, max_length=160)
    reason: str = Field(min_length=10, max_length=500)
    expected_source: str = Field(min_length=2, max_length=120)
    priority: MissingEvidencePriority


class CustomerMessageDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str = Field(min_length=3, max_length=140)
    body: str = Field(min_length=20, max_length=1500)
    tone: Literal["NEUTRAL", "EMPATHETIC", "URGENT"]
    send_status: Literal["DRAFT_ONLY"] = "DRAFT_ONLY"


class EvidencePack(BaseModel):
    """Schema-constrained output produced by AI or fallback logic."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    payment_id: str = Field(min_length=4, max_length=100)
    case_summary: str = Field(min_length=20, max_length=1000)
    case_strength: CaseStrength
    recommended_action: Literal[
        "MONITOR",
        "PREPARE_EVIDENCE",
        "MANUAL_REVIEW",
        "RECOMMEND_REFUND",
    ]
    action_rationale: str = Field(min_length=10, max_length=700)
    evidence_items: list[EvidenceItem] = Field(
        min_length=1,
        max_length=10,
    )
    missing_evidence: list[MissingEvidenceItem] = Field(
        default_factory=list,
        max_length=10,
    )
    customer_message: CustomerMessageDraft | None = None
    limitations: list[str] = Field(
        min_length=1,
        max_length=8,
    )
    human_approval_required: Literal[True] = True
    action_executed: Literal[False] = False


class EvidenceContractError(ValueError):
    pass


def validate_pack_against_case(
    evidence_pack: EvidencePack,
    evidence_case: EvidenceCase,
) -> None:
    """Enforce invariants the language model is not allowed to change."""
    errors: list[str] = []

    if evidence_pack.payment_id != evidence_case.payment_id:
        errors.append("payment_id does not match the evidence case")

    if (
        evidence_pack.recommended_action
        != evidence_case.deterministic_recommended_action
    ):
        errors.append(
            "recommended_action conflicts with the deterministic policy"
        )

    allowed_fact_ids = {
        fact.fact_id
        for fact in evidence_case.facts
    }

    cited_fact_ids = {
        fact_id
        for item in evidence_pack.evidence_items
        for fact_id in item.citation_fact_ids
    }

    unknown_fact_ids = cited_fact_ids - allowed_fact_ids

    if unknown_fact_ids:
        errors.append(
            "unknown citation fact IDs: "
            + ", ".join(sorted(unknown_fact_ids))
        )

    if errors:
        raise EvidenceContractError("; ".join(errors))
