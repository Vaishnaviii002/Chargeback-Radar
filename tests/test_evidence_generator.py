from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.evidence_contract import (
    CaseStrength,
    EvidenceCase,
    EvidenceContractError,
    EvidenceFact,
    EvidenceItem,
    EvidencePack,
    EvidenceStrength,
    FactSource,
)
from src.evidence_generator import (
    EvidenceGenerationDisabledError,
    EvidenceGenerationError,
    EvidenceGenerationIncompleteError,
    EvidenceGenerationRefusedError,
    EvidenceGeneratorConfig,
    OpenAIEvidenceGenerator,
    build_evidence_messages,
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
                fact_id="SHIPMENT_PROMISED_AT",
                source=FactSource.SHIPMENT,
                label="Promised shipment deadline",
                value="2026-01-10T00:00:00+00:00",
                observed_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
        ],
    )


def make_pack() -> EvidencePack:
    return EvidencePack(
        payment_id="pay_test_001",
        case_summary=(
            "The promised shipment date has passed for this payment."
        ),
        case_strength=CaseStrength.MODERATE,
        recommended_action="PREPARE_EVIDENCE",
        action_rationale=(
            "Prepare the available fulfilment records for human review."
        ),
        evidence_items=[
            EvidenceItem(
                title="Shipment promise",
                statement=(
                    "The promised shipment deadline was 10 January 2026."
                ),
                citation_fact_ids=["SHIPMENT_PROMISED_AT"],
                strength=EvidenceStrength.STRONG,
                relevance=(
                    "This timestamp is relevant to a possible "
                    "non-delivery dispute."
                ),
            )
        ],
        missing_evidence=[],
        customer_message=None,
        limitations=[
            "The risk estimate comes from a synthetic-data model."
        ],
        human_approval_required=True,
        action_executed=False,
    )


class FakeResponses:
    def __init__(self, response) -> None:
        self.response = response
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response) -> None:
        self.responses = FakeResponses(response)


def config(*, enabled: bool = True) -> EvidenceGeneratorConfig:
    return EvidenceGeneratorConfig(
        model="gpt-5.6",
        timeout_seconds=20,
        max_retries=1,
        enabled=enabled,
        api_key="test-key-not-used-by-fake-client",
    )


def successful_response(pack: EvidencePack):
    return SimpleNamespace(
        id="resp_test_001",
        model="gpt-5.6-2026-08-07",
        status="completed",
        output=[],
        output_parsed=pack,
    )


def test_prompt_contains_only_validated_case_envelope() -> None:
    messages = build_evidence_messages(make_case())

    assert [message["role"] for message in messages] == [
        "developer",
        "user",
    ]
    assert "GROUND_TRUTH_CASE_JSON" in messages[1]["content"]
    assert "pay_test_001" in messages[1]["content"]
    assert "Never invent" in messages[0]["content"]


def test_calls_responses_parse_with_pydantic_schema() -> None:
    client = FakeClient(successful_response(make_pack()))
    generator = OpenAIEvidenceGenerator(
        client=client,
        config=config(),
    )

    result = generator.generate(make_case())
    call = client.responses.calls[0]

    assert call["model"] == "gpt-5.6"
    assert call["text_format"] is EvidencePack
    assert call["input"][0]["role"] == "developer"
    assert result.evidence_pack.payment_id == "pay_test_001"
    assert result.response_id == "resp_test_001"
    assert result.provider == "openai"


def test_changed_deterministic_action_is_rejected() -> None:
    invalid_pack = make_pack().model_copy(
        update={"recommended_action": "RECOMMEND_REFUND"}
    )
    client = FakeClient(successful_response(invalid_pack))

    with pytest.raises(EvidenceContractError, match="deterministic"):
        OpenAIEvidenceGenerator(
            client=client,
            config=config(),
        ).generate(make_case())


def test_invented_citation_is_rejected() -> None:
    pack = make_pack()
    invalid_item = pack.evidence_items[0].model_copy(
        update={"citation_fact_ids": ["INVENTED_FACT"]}
    )
    invalid_pack = pack.model_copy(
        update={"evidence_items": [invalid_item]}
    )
    client = FakeClient(successful_response(invalid_pack))

    with pytest.raises(EvidenceContractError, match="INVENTED_FACT"):
        OpenAIEvidenceGenerator(
            client=client,
            config=config(),
        ).generate(make_case())


def test_refusal_is_detected() -> None:
    response = SimpleNamespace(
        status="completed",
        output=[
            SimpleNamespace(
                content=[
                    SimpleNamespace(
                        refusal="I cannot process this request."
                    )
                ]
            )
        ],
        output_parsed=None,
    )

    with pytest.raises(EvidenceGenerationRefusedError, match="refused"):
        OpenAIEvidenceGenerator(
            client=FakeClient(response),
            config=config(),
        ).generate(make_case())


def test_incomplete_response_is_detected() -> None:
    response = SimpleNamespace(
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        output=[],
        output_parsed=None,
    )

    with pytest.raises(
        EvidenceGenerationIncompleteError,
        match="max_output_tokens",
    ):
        OpenAIEvidenceGenerator(
            client=FakeClient(response),
            config=config(),
        ).generate(make_case())


def test_missing_parsed_output_is_detected() -> None:
    response = SimpleNamespace(
        status="completed",
        output=[],
        output_parsed=None,
    )

    with pytest.raises(
        EvidenceGenerationIncompleteError,
        match="no parsed evidence pack",
    ):
        OpenAIEvidenceGenerator(
            client=FakeClient(response),
            config=config(),
        ).generate(make_case())


def test_disabled_generator_never_calls_api() -> None:
    client = FakeClient(successful_response(make_pack()))

    with pytest.raises(EvidenceGenerationDisabledError, match="disabled"):
        OpenAIEvidenceGenerator(
            client=client,
            config=config(enabled=False),
        ).generate(make_case())

    assert client.responses.calls == []


def test_missing_key_has_safe_configuration_error() -> None:
    generator = OpenAIEvidenceGenerator(
        config=EvidenceGeneratorConfig(
            api_key=None,
        )
    )

    with pytest.raises(EvidenceGenerationError, match="OPENAI_API_KEY"):
        generator.generate(make_case())
