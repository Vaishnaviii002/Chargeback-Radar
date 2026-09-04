from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from src.explanation_text import DISCLAIMER
from src.model_explanation_generator import (
    AIExplanationDraft,
    EXPECTED_LABEL,
    ModelExplanationGenerationDisabledError,
    ModelExplanationGenerationError,
    ModelExplanationGenerationIncompleteError,
    ModelExplanationGenerationRefusedError,
    ModelExplanationGenerationUnsafeError,
    ModelExplanationGeneratorConfig,
    OpenAIModelExplanationGenerator,
    build_phrasing_messages,
    prepare_factors,
    validate_ai_draft,
)
from src.model_explanation_service import (
    ExplanationFactor,
)


class FakeResponses:
    def __init__(
        self,
        *,
        response: Any = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def parse(
        self,
        **kwargs: Any,
    ) -> Any:
        self.calls.append(kwargs)

        if self.error is not None:
            raise self.error

        return self.response


class FakeClient:
    def __init__(
        self,
        *,
        response: Any = None,
        error: Exception | None = None,
    ) -> None:
        self.responses = FakeResponses(
            response=response,
            error=error,
        )


def _positive_factors() -> list[
    ExplanationFactor
]:
    return [
        ExplanationFactor(
            feature="device_is_new",
            value=1,
            shap_value=0.42,
            direction="increases_risk",
        ),
        ExplanationFactor(
            feature="txns_last_24h",
            value=5,
            shap_value=0.17,
            direction="increases_risk",
        ),
        ExplanationFactor(
            feature="amount_paise",
            value=125000,
            shap_value=0.08,
            direction="increases_risk",
        ),
        ExplanationFactor(
            feature="is_digital_good",
            value=1,
            shap_value=0.03,
            direction="increases_risk",
        ),
    ]


def _negative_factors() -> list[
    ExplanationFactor
]:
    return [
        ExplanationFactor(
            feature="email_verified",
            value=1,
            shap_value=-0.25,
            direction="decreases_risk",
        ),
        ExplanationFactor(
            feature="phone_verified",
            value=1,
            shap_value=-0.06,
            direction="decreases_risk",
        ),
        ExplanationFactor(
            feature="has_prior_order",
            value=1,
            shap_value=-0.02,
            direction="decreases_risk",
        ),
    ]


def _valid_draft(
    **changes: Any,
) -> AIExplanationDraft:
    payload: dict[str, Any] = {
        "summary": (
            "The supplied inputs contributed to "
            "the model's risk estimate in "
            "opposite directions."
        ),
        "referenced_positive_features": [
            "device_is_new",
            "txns_last_24h",
            "amount_paise",
        ],
        "referenced_negative_features": [
            "email_verified",
            "phone_verified",
        ],
        "label": EXPECTED_LABEL,
        "limitation": DISCLAIMER,
    }

    payload.update(changes)

    return AIExplanationDraft.model_validate(
        payload
    )


def _completed_response(
    draft: AIExplanationDraft | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        status="completed",
        output=[],
        output_parsed=(
            draft
            if draft is not None
            else _valid_draft()
        ),
        model="gpt-test",
        id="resp_test_001",
    )


def _enabled_config(
) -> ModelExplanationGeneratorConfig:
    return ModelExplanationGeneratorConfig(
        model="gpt-test",
        timeout_seconds=5,
        max_retries=0,
        enabled=True,
        api_key=None,
    )


def test_prepare_factors_enforces_display_limits() -> None:
    positive, negative = prepare_factors(
        _positive_factors(),
        _negative_factors(),
    )

    assert [
        factor.feature
        for factor in positive
    ] == [
        "device_is_new",
        "txns_last_24h",
        "amount_paise",
    ]

    assert [
        factor.feature
        for factor in negative
    ] == [
        "email_verified",
        "phone_verified",
    ]


def test_messages_contain_only_factor_envelope() -> None:
    messages = build_phrasing_messages(
        _positive_factors(),
        _negative_factors(),
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "developer"
    assert messages[1]["role"] == "user"

    serialized = str(messages)

    assert "TRUSTED_MODEL_FACTORS_JSON" in serialized
    assert "device_is_new" in serialized
    assert "email_verified" in serialized
    assert EXPECTED_LABEL in serialized
    assert DISCLAIMER in messages[1]["content"]

    assert "payment_id" not in serialized
    assert "customer_id" not in serialized
    assert "customer_name" not in serialized
    assert "customer_email" not in serialized
    assert "dispute_id" not in serialized
    assert "chargeback_within_120d" not in serialized
    assert "true_fraud" not in serialized


def test_valid_openai_result_is_accepted() -> None:
    client = FakeClient(
        response=_completed_response()
    )

    generator = OpenAIModelExplanationGenerator(
        client=client,
        config=_enabled_config(),
    )

    result = generator.generate(
        positive_factors=_positive_factors(),
        negative_factors=_negative_factors(),
    )

    assert result.provider == "openai"
    assert result.model == "gpt-test"
    assert result.response_id == "resp_test_001"
    assert result.latency_ms >= 0
    assert result.draft == _valid_draft()

    assert len(
        client.responses.calls
    ) == 1

    call = client.responses.calls[0]

    assert call["model"] == "gpt-test"
    assert (
        call["text_format"]
        is AIExplanationDraft
    )


def test_disabled_generation_is_rejected() -> None:
    generator = OpenAIModelExplanationGenerator(
        client=FakeClient(),
        config=ModelExplanationGeneratorConfig(
            model="gpt-test",
            timeout_seconds=5,
            max_retries=0,
            enabled=False,
            api_key=None,
        ),
    )

    with pytest.raises(
        ModelExplanationGenerationDisabledError,
        match="MODEL_EXPLANATION_AI_ENABLED",
    ):
        generator.generate(
            positive_factors=_positive_factors(),
            negative_factors=_negative_factors(),
        )


def test_missing_api_key_is_rejected() -> None:
    generator = OpenAIModelExplanationGenerator(
        config=_enabled_config(),
    )

    with pytest.raises(
        ModelExplanationGenerationError,
        match="OPENAI_API_KEY is missing",
    ):
        generator._client_or_create()


def test_provider_error_is_wrapped() -> None:
    client = FakeClient(
        error=RuntimeError(
            "provider unavailable"
        )
    )

    generator = OpenAIModelExplanationGenerator(
        client=client,
        config=_enabled_config(),
    )

    with pytest.raises(
        ModelExplanationGenerationError,
        match=(
            "OpenAI model-explanation "
            "generation failed"
        ),
    ):
        generator.generate(
            positive_factors=_positive_factors(),
            negative_factors=_negative_factors(),
        )


def test_model_refusal_is_rejected() -> None:
    response = SimpleNamespace(
        status="completed",
        output=[
            {
                "content": [
                    {
                        "refusal": (
                            "Unable to comply"
                        )
                    }
                ]
            }
        ],
        output_parsed=None,
        model="gpt-test",
        id="resp_refusal",
    )

    generator = OpenAIModelExplanationGenerator(
        client=FakeClient(
            response=response
        ),
        config=_enabled_config(),
    )

    with pytest.raises(
        ModelExplanationGenerationRefusedError,
        match="refused",
    ):
        generator.generate(
            positive_factors=_positive_factors(),
            negative_factors=_negative_factors(),
        )


def test_incomplete_response_is_rejected() -> None:
    response = SimpleNamespace(
        status="incomplete",
        incomplete_details=SimpleNamespace(
            reason="max_output_tokens"
        ),
        output=[],
        output_parsed=None,
        model="gpt-test",
        id="resp_incomplete",
    )

    generator = OpenAIModelExplanationGenerator(
        client=FakeClient(
            response=response
        ),
        config=_enabled_config(),
    )

    with pytest.raises(
        ModelExplanationGenerationIncompleteError,
        match="max_output_tokens",
    ):
        generator.generate(
            positive_factors=_positive_factors(),
            negative_factors=_negative_factors(),
        )


def test_missing_parsed_output_is_rejected() -> None:
    response = SimpleNamespace(
        status="completed",
        output=[],
        output_parsed=None,
        model="gpt-test",
        id="resp_missing",
    )

    generator = OpenAIModelExplanationGenerator(
        client=FakeClient(
            response=response
        ),
        config=_enabled_config(),
    )

    with pytest.raises(
        ModelExplanationGenerationIncompleteError,
        match="no parsed explanation draft",
    ):
        generator.generate(
            positive_factors=_positive_factors(),
            negative_factors=_negative_factors(),
        )


def test_changed_positive_factors_are_rejected() -> None:
    draft = _valid_draft(
        referenced_positive_features=[
            "device_is_new",
            "amount_paise",
            "txns_last_24h",
        ]
    )

    with pytest.raises(
        ModelExplanationGenerationUnsafeError,
        match="positive factors",
    ):
        validate_ai_draft(
            draft,
            positive_factors=_positive_factors(),
            negative_factors=_negative_factors(),
        )


def test_added_factor_is_rejected() -> None:
    draft = _valid_draft(
        referenced_negative_features=[
            "email_verified",
            "ip_country_matches_billing",
        ]
    )

    with pytest.raises(
        ModelExplanationGenerationUnsafeError,
        match="negative factors",
    ):
        validate_ai_draft(
            draft,
            positive_factors=_positive_factors(),
            negative_factors=_negative_factors(),
        )


def test_changed_limitation_is_rejected() -> None:
    draft = _valid_draft(
        limitation=(
            "This explanation proves the outcome."
        )
    )

    with pytest.raises(
        ModelExplanationGenerationUnsafeError,
        match="required limitation",
    ):
        validate_ai_draft(
            draft,
            positive_factors=_positive_factors(),
            negative_factors=_negative_factors(),
        )


@pytest.mark.parametrize(
    "unsafe_summary",
    [
        (
            "The supplied inputs prove that fraud "
            "occurred according to the model."
        ),
        (
            "The supplied inputs caused the model "
            "risk estimate to increase."
        ),
        (
            "The model demonstrated customer intent "
            "from the supplied inputs."
        ),
        (
            "The model recommends a refund based "
            "on the supplied inputs."
        ),
        (
            "The model considers this dispute "
            "evidence from the supplied inputs."
        ),
    ],
)
def test_unsafe_language_is_rejected(
    unsafe_summary: str,
) -> None:
    draft = _valid_draft(
        summary=unsafe_summary
    )

    with pytest.raises(
        ModelExplanationGenerationUnsafeError,
        match="unsafe",
    ):
        validate_ai_draft(
            draft,
            positive_factors=_positive_factors(),
            negative_factors=_negative_factors(),
        )


def test_numeric_claim_is_rejected() -> None:
    draft = _valid_draft(
        summary=(
            "The supplied inputs contributed to "
            "the model estimate by 42 percent."
        )
    )

    with pytest.raises(
        ModelExplanationGenerationUnsafeError,
        match="numeric claim",
    ):
        validate_ai_draft(
            draft,
            positive_factors=_positive_factors(),
            negative_factors=_negative_factors(),
        )


def test_summary_must_identify_model_context() -> None:
    draft = _valid_draft(
        summary=(
            "The supplied inputs contributed in "
            "opposite directions overall."
        )
    )

    with pytest.raises(
        ModelExplanationGenerationUnsafeError,
        match="model context",
    ):
        validate_ai_draft(
            draft,
            positive_factors=_positive_factors(),
            negative_factors=_negative_factors(),
        )


def test_wrong_input_direction_is_rejected() -> None:
    invalid_positive = [
        ExplanationFactor(
            feature="device_is_new",
            value=1,
            shap_value=0.42,
            direction="decreases_risk",
        )
    ]

    with pytest.raises(
        ModelExplanationGenerationUnsafeError,
        match="wrong direction list",
    ):
        prepare_factors(
            invalid_positive,
            _negative_factors(),
        )


def test_forbidden_input_feature_is_rejected() -> None:
    forbidden_positive = [
        ExplanationFactor(
            feature="customer_id",
            value="cust_secret",
            shap_value=0.42,
            direction="increases_risk",
        )
    ]

    with pytest.raises(
        ModelExplanationGenerationUnsafeError,
        match="forbidden field",
    ):
        prepare_factors(
            forbidden_positive,
            _negative_factors(),
        )


def test_unordered_input_factors_are_rejected() -> None:
    positive = _positive_factors()
    positive[0], positive[1] = (
        positive[1],
        positive[0],
    )

    with pytest.raises(
        ModelExplanationGenerationUnsafeError,
        match="not ordered",
    ):
        prepare_factors(
            positive,
            _negative_factors(),
        )


def test_draft_rejects_unknown_fields() -> None:
    payload = _valid_draft().model_dump()
    payload["unsupported_reason"] = "invented"

    with pytest.raises(ValidationError):
        AIExplanationDraft.model_validate(
            payload
        )