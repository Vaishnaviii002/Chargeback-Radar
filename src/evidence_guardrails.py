from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field

from src.evidence_contract import (
    EvidenceCase,
    EvidencePack,
    EvidenceStrength,
    FactSource,
    validate_pack_against_case,
)


PROMPT_INJECTION_PATTERNS = (
    r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\b",
    r"\b(?:system|developer)\s+(?:prompt|message|instructions?)\b",
    r"\b(?:reveal|print|show)\s+(?:the\s+)?(?:prompt|instructions?)\b",
    r"\bjailbreak\b",
    r"\bdo\s+not\s+follow\s+(?:the\s+)?instructions?\b",
    r"<\s*/?\s*(?:system|assistant|developer)\s*>",
)

ACCUSATORY_PATTERNS = (
    r"\bfraudster\b",
    r"\bscammer\b",
    r"\bcustomer\s+(?:committed|attempted|carried\s+out)\s+fraud\b",
    r"\b(?:confirmed|proven|definite)\s+fraud\b",
    r"\bstolen\s+card\b",
)

EXECUTED_ACTION_PATTERNS = (
    r"\bwe\s+(?:have\s+)?(?:issued|processed|completed)\s+(?:the\s+)?refund\b",
    r"\brefund\s+(?:has\s+been|was)\s+(?:issued|processed|completed)\b",
    r"\byour\s+refund\s+is\s+(?:complete|completed|processed)\b",
    r"\breview\s+(?:has\s+been|is)\s+completed\b",
    r"\bdispute\s+(?:has\s+been|is)\s+(?:won|resolved|closed)\b",
)

EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    flags=re.IGNORECASE,
)
INDIAN_PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+91[-\s]?)?[6-9]\d{9}(?!\d)"
)
CARD_CANDIDATE_PATTERN = re.compile(
    r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"
)
NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z])(?:INR\s*|₹\s*)?"
    r"\d[\d,]*(?:\.\d+)?%?(?![A-Za-z])",
    flags=re.IGNORECASE,
)
ISO_DATE_PATTERN = re.compile(
    r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)"
)
WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_'-]{2,}")

STOP_WORDS = {
    "and",
    "are",
    "for",
    "from",
    "has",
    "have",
    "into",
    "that",
    "the",
    "their",
    "this",
    "was",
    "were",
    "with",
    "without",
    "payment",
    "recorded",
    "available",
}

OPERATIONAL_SOURCES = {
    FactSource.PAYMENT,
    FactSource.CAPTURE_RULE,
    FactSource.LIFECYCLE_RULE,
    FactSource.ORDER,
    FactSource.SHIPMENT,
    FactSource.REFUND,
}


class GuardrailViolation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{2,63}$")
    field_path: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=5, max_length=500)


class GuardrailReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    checks_run: int = Field(ge=1)
    facts_checked: int = Field(ge=0)
    evidence_items_checked: int = Field(ge=0)
    citations_checked: int = Field(ge=0)
    violations: list[GuardrailViolation] = Field(default_factory=list)


class EvidenceGuardrailError(ValueError):
    def __init__(self, violations: list[GuardrailViolation]) -> None:
        self.violations = violations
        detail = "; ".join(
            f"{violation.code} at {violation.field_path}: "
            f"{violation.message}"
            for violation in violations
        )
        super().__init__(f"Evidence guardrail blocked the request: {detail}")


def _violation(
    code: str,
    field_path: str,
    message: str,
) -> GuardrailViolation:
    return GuardrailViolation(
        code=code,
        field_path=field_path,
        message=message,
    )


def _matches_any(text: str, patterns: Iterable[str]) -> bool:
    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in patterns
    )


def _luhn_valid(candidate: str) -> bool:
    digits = [int(character) for character in candidate if character.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False

    checksum = 0
    parity = len(digits) % 2

    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit

    return checksum % 10 == 0


def _contains_card_number(text: str) -> bool:
    return any(
        _luhn_valid(match.group(0))
        for match in CARD_CANDIDATE_PATTERN.finditer(text)
    )


def _normalise_number(raw_value: str) -> tuple[str, bool] | None:
    is_percentage = raw_value.strip().endswith("%")
    cleaned = re.sub(
        r"(?:INR|₹|,|\s|%)",
        "",
        raw_value,
        flags=re.IGNORECASE,
    )

    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None

    normalised = format(value.normalize(), "f")
    if "." in normalised:
        normalised = normalised.rstrip("0").rstrip(".")

    return normalised, is_percentage


def _numeric_claims(text: str) -> set[tuple[str, bool]]:
    claims: set[tuple[str, bool]] = set()

    # Make each ISO date component comparable with natural-language dates,
    # for example 2026-01-10 and "10 January 2026".
    for date_match in ISO_DATE_PATTERN.finditer(text):
        for component in date_match.groups():
            normalised = _normalise_number(component)
            if normalised is not None:
                claims.add(normalised)

    for match in NUMBER_PATTERN.finditer(text):
        normalised = _normalise_number(match.group(0))
        if normalised is not None:
            claims.add(normalised)

    return claims


def _keywords(text: str) -> set[str]:
    return {
        word.lower()
        for word in WORD_PATTERN.findall(text)
        if word.lower() not in STOP_WORDS
    }


def _all_generated_text(pack: EvidencePack) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = [
        ("case_summary", pack.case_summary),
        ("action_rationale", pack.action_rationale),
    ]

    for index, item in enumerate(pack.evidence_items):
        values.extend(
            [
                (f"evidence_items.{index}.title", item.title),
                (f"evidence_items.{index}.statement", item.statement),
                (f"evidence_items.{index}.relevance", item.relevance),
            ]
        )

    for index, item in enumerate(pack.missing_evidence):
        values.extend(
            [
                (f"missing_evidence.{index}.item", item.item),
                (f"missing_evidence.{index}.reason", item.reason),
                (
                    f"missing_evidence.{index}.expected_source",
                    item.expected_source,
                ),
            ]
        )

    for index, limitation in enumerate(pack.limitations):
        values.append((f"limitations.{index}", limitation))

    if pack.customer_message is not None:
        values.extend(
            [
                ("customer_message.subject", pack.customer_message.subject),
                ("customer_message.body", pack.customer_message.body),
            ]
        )

    return values


def _case_safety_violations(
    evidence_case: EvidenceCase,
) -> list[GuardrailViolation]:
    violations: list[GuardrailViolation] = []

    for index, fact in enumerate(evidence_case.facts):
        field_path = f"facts.{index}.value"
        value = fact.value

        if _matches_any(value, PROMPT_INJECTION_PATTERNS):
            violations.append(
                _violation(
                    "PROMPT_INJECTION_INPUT",
                    field_path,
                    "The fact value contains instruction-like text.",
                )
            )

        if EMAIL_PATTERN.search(value):
            violations.append(
                _violation(
                    "EMAIL_PII_INPUT",
                    field_path,
                    "An email address must not be sent to the drafting model.",
                )
            )

        if INDIAN_PHONE_PATTERN.search(value):
            violations.append(
                _violation(
                    "PHONE_PII_INPUT",
                    field_path,
                    "A phone number must not be sent to the drafting model.",
                )
            )

        if _contains_card_number(value):
            violations.append(
                _violation(
                    "CARD_PII_INPUT",
                    field_path,
                    "A possible payment-card number was detected.",
                )
            )

    return violations


def validate_evidence_case_safety(
    evidence_case: EvidenceCase,
) -> GuardrailReport:
    violations = _case_safety_violations(evidence_case)

    if violations:
        raise EvidenceGuardrailError(violations)

    return GuardrailReport(
        passed=True,
        checks_run=4,
        facts_checked=len(evidence_case.facts),
        evidence_items_checked=0,
        citations_checked=0,
        violations=[],
    )


def validate_generated_evidence_pack(
    evidence_pack: EvidencePack,
    evidence_case: EvidenceCase,
) -> GuardrailReport:
    # Existing contract checks are part of the blocking output boundary.
    validate_pack_against_case(evidence_pack, evidence_case)

    violations = _case_safety_violations(evidence_case)
    facts_by_id = {
        fact.fact_id: fact
        for fact in evidence_case.facts
    }
    all_fact_text = " ".join(
        f"{fact.label} {fact.value}"
        for fact in evidence_case.facts
    )
    all_case_numbers = _numeric_claims(all_fact_text)

    for field_path, value in _all_generated_text(evidence_pack):
        if _matches_any(value, ACCUSATORY_PATTERNS):
            violations.append(
                _violation(
                    "ACCUSATORY_LANGUAGE",
                    field_path,
                    "The draft asserts wrongdoing that the supplied facts do not prove.",
                )
            )

        if field_path in {
            "case_summary",
            "action_rationale",
            "customer_message.subject",
            "customer_message.body",
        }:
            unsupported = _numeric_claims(value) - all_case_numbers
            if unsupported:
                violations.append(
                    _violation(
                        "UNSUPPORTED_NUMERIC_CLAIM",
                        field_path,
                        "A number, amount, percentage, or date is absent from the case facts.",
                    )
                )

    if evidence_pack.customer_message is not None:
        customer_text = (
            f"{evidence_pack.customer_message.subject} "
            f"{evidence_pack.customer_message.body}"
        )
        if _matches_any(customer_text, EXECUTED_ACTION_PATTERNS):
            violations.append(
                _violation(
                    "CUSTOMER_MESSAGE_EXECUTION_CLAIM",
                    "customer_message",
                    "The draft claims that a protected action has already executed.",
                )
            )

    for index, item in enumerate(evidence_pack.evidence_items):
        cited_facts = [
            facts_by_id[fact_id]
            for fact_id in item.citation_fact_ids
            if fact_id in facts_by_id
        ]
        cited_text = " ".join(
            f"{fact.label} {fact.value}"
            for fact in cited_facts
        )

        unsupported_numbers = (
            _numeric_claims(item.statement)
            - _numeric_claims(cited_text)
        )
        if unsupported_numbers:
            violations.append(
                _violation(
                    "CITATION_NUMERIC_MISMATCH",
                    f"evidence_items.{index}.statement",
                    "The statement contains a number or date not present in its cited facts.",
                )
            )

        statement_keywords = _keywords(
            f"{item.title} {item.statement}"
        )
        cited_keywords = _keywords(cited_text)
        if statement_keywords and not (
            statement_keywords.intersection(cited_keywords)
        ):
            violations.append(
                _violation(
                    "CITATION_SEMANTIC_MISMATCH",
                    f"evidence_items.{index}",
                    "The statement has no meaningful lexical overlap with its cited facts.",
                )
            )

        if (
            item.strength == EvidenceStrength.STRONG
            and cited_facts
            and not any(
                fact.source in OPERATIONAL_SOURCES
                for fact in cited_facts
            )
        ):
            violations.append(
                _violation(
                    "MODEL_ONLY_EVIDENCE_OVERSTATED",
                    f"evidence_items.{index}.strength",
                    "Model or policy output alone cannot be labelled strong evidence.",
                )
            )

    limitations_text = " ".join(evidence_pack.limitations).lower()
    if not any(
        term in limitations_text
        for term in ("synthetic", "model", "estimate", "uncertain")
    ):
        violations.append(
            _violation(
                "MODEL_LIMITATION_MISSING",
                "limitations",
                "The draft must disclose model or synthetic-data uncertainty.",
            )
        )

    if violations:
        raise EvidenceGuardrailError(violations)

    return GuardrailReport(
        passed=True,
        checks_run=10,
        facts_checked=len(evidence_case.facts),
        evidence_items_checked=len(evidence_pack.evidence_items),
        citations_checked=sum(
            len(item.citation_fact_ids)
            for item in evidence_pack.evidence_items
        ),
        violations=[],
    )
