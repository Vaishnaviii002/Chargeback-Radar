from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from src.explanation_text import (
    DISCLAIMER,
    build_deterministic_explanation,
)
from src.model_explanation_delivery import (
    OPENAI_PHRASING_VERSION,
    ModelExplanationDeliveryService,
    deliver_model_explanation,
)
from src.model_explanation_generator import (
    AIExplanationDraft,
    EXPECTED_LABEL,
    ModelExplanationGenerationDisabledError,
    ModelExplanationGenerationError,
    ModelExplanationGenerationIncompleteError,
    ModelExplanationGenerationRefusedError,
    ModelExplanationGenerationResult,
    ModelExplanationGenerationUnsafeError,
)
from src.model_explanation_service import (
    ExplanationFactor,
    ModelExplanation,
)


class StaticGenerator:
    def __init__(
        self,
        *,
        result: (
            ModelExplanationGenerationResult
            | None
        ) = None,
        error: BaseException | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[
            dict[str, Any]
        ] = []

    def generate(
        self,
        **kwargs: Any,
    ) -> ModelExplanationGenerationResult:
        self.calls.append(kwargs)

        if self.error is not None:
            raise self.error

        assert self.result is not None
        return self.result


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
    ]


def _source_explanation(
    *,
    disclaimer: str = DISCLAIMER,
    explanation_text: str | None = None,
) -> ModelExplanation:
    positive = _positive_factors()
    negative = _negative_factors()

    deterministic_text = (
        build_deterministic_explanation(
            positive_factors=[
                factor.model_dump()
                for factor in positive
            ],
            negative_factors=[
                factor.model_dump()
                for factor in negative
            ],
            max_positive=3,
            max_negative=2,
        )
    )

    return ModelExplanation(
        payment_id="pay_test_001",
        model_version="0.1.0",
        shap_explanation_version="shap-v1",
        text_explanation_version="plain-v1",
        delivery_mode="deterministic",
        label=EXPECTED_LABEL,
        explanation=(
            explanation_text
            if explanation_text is not None
            else deterministic_text
        ),
        disclaimer=disclaimer,
        positive_factors=positive,
        negative_factors=negative,
    )


def _live_generation(
) -> ModelExplanationGenerationResult:
    draft = AIExplanationDraft(
        summary=(
            "The supplied inputs contributed to "
            "the model's risk estimate in "
            "opposite directions."
        ),
        referenced_positive_features=[
            "device_is_new",
            "txns_last_24h",
            "amount_paise",
        ],
        referenced_negative_features=[
            "email_verified",
            "phone_verified",
        ],
        label=EXPECTED_LABEL,
        limitation=DISCLAIMER,
    )

    return ModelExplanationGenerationResult(
        draft=draft,
        provider="openai",
        model="gpt-test",
        response_id="resp_test_001",
        latency_ms=14,
    )


def test_live_openai_delivery_preserves_source() -> None:
    source = _source_explanation()

    generator = StaticGenerator(
        result=_live_generation()
    )

    service = (
        ModelExplanationDeliveryService(
            generator=generator
        )
    )

    result = service.deliver(
        source
    )

    assert result.payment_id == source.payment_id
    assert (
        result.model_version
        == source.model_version
    )
    assert (
        result.shap_explanation_version
        == source.shap_explanation_version
    )
    assert (
        result.deterministic_text_version
        == source.text_explanation_version
    )
    assert (
        result.delivery_text_version
        == OPENAI_PHRASING_VERSION
    )

    assert result.delivery_mode == "LIVE_OPENAI"
    assert result.provider == "openai"
    assert result.model == "gpt-test"
    assert result.response_id == "resp_test_001"
    assert result.latency_ms == 14
    assert result.fallback_used is False
    assert result.fallback_reason is None

    assert result.label == EXPECTED_LABEL
    assert result.disclaimer == DISCLAIMER
    assert result.explanation.endswith(
        source.explanation
    )
    assert (
        result.explanation.count(
            DISCLAIMER
        )
        == 1
    )

    assert (
        result.positive_factors
        == source.positive_factors
    )
    assert (
        result.negative_factors
        == source.negative_factors
    )

    assert len(generator.calls) == 1
    assert (
        generator.calls[0][
            "positive_factors"
        ]
        == source.positive_factors
    )
    assert (
        generator.calls[0][
            "negative_factors"
        ]
        == source.negative_factors
    )


@pytest.mark.parametrize(
    ("error", "expected_reason"),
    [
        (
            ModelExplanationGenerationDisabledError(
                "disabled"
            ),
            "AI_DISABLED",
        ),
        (
            ModelExplanationGenerationRefusedError(
                "refused"
            ),
            "MODEL_REFUSAL",
        ),
        (
            ModelExplanationGenerationIncompleteError(
                "incomplete"
            ),
            "INCOMPLETE_RESPONSE",
        ),
        (
            ModelExplanationGenerationUnsafeError(
                "unsafe"
            ),
            "OUTPUT_GUARDRAIL_BLOCK",
        ),
        (
            ModelExplanationGenerationError(
                "provider failed"
            ),
            "API_GENERATION_FAILED",
        ),
    ],
)
def test_recoverable_error_uses_fallback(
    error: BaseException,
    expected_reason: str,
) -> None:
    source = _source_explanation()

    service = (
        ModelExplanationDeliveryService(
            generator=StaticGenerator(
                error=error
            )
        )
    )

    result = service.deliver(
        source
    )

    assert (
        result.delivery_mode
        == "DETERMINISTIC_FALLBACK"
    )
    assert (
        result.provider
        == "deterministic_formatter"
    )
    assert result.model is None
    assert result.latency_ms == 0
    assert result.fallback_used is True
    assert (
        result.fallback_reason
        == expected_reason
    )

    assert (
        result.delivery_text_version
        == source.text_explanation_version
    )
    assert (
        result.explanation
        == source.explanation
    )
    assert result.label == EXPECTED_LABEL
    assert result.disclaimer == DISCLAIMER
    assert (
        result.positive_factors
        == source.positive_factors
    )
    assert (
        result.negative_factors
        == source.negative_factors
    )


def test_schema_error_uses_fallback() -> None:
    with pytest.raises(
        ValidationError,
    ) as captured:
        AIExplanationDraft.model_validate(
            {}
        )

    source = _source_explanation()

    service = (
        ModelExplanationDeliveryService(
            generator=StaticGenerator(
                error=captured.value
            )
        )
    )

    result = service.deliver(
        source
    )

    assert result.fallback_used is True
    assert (
        result.fallback_reason
        == "OUTPUT_SCHEMA_INVALID"
    )
    assert (
        result.explanation
        == source.explanation
    )


def test_api_timeout_category_is_safe() -> None:
    class APITimeoutError(Exception):
        request_id = "req_timeout_001"

    try:
        raise APITimeoutError(
            "secret provider details"
        )
    except APITimeoutError as cause:
        wrapped = (
            ModelExplanationGenerationError(
                "generation failed"
            )
        )
        wrapped.__cause__ = cause

    source = _source_explanation()

    service = (
        ModelExplanationDeliveryService(
            generator=StaticGenerator(
                error=wrapped
            )
        )
    )

    result = service.deliver(
        source
    )

    assert result.fallback_used is True
    assert (
        result.fallback_reason
        == "API_TIMEOUT"
    )
    assert (
        result.response_id
        == "req_timeout_001"
    )


def test_programming_error_is_not_hidden() -> None:
    service = (
        ModelExplanationDeliveryService(
            generator=StaticGenerator(
                error=RuntimeError(
                    "programming bug"
                )
            )
        )
    )

    with pytest.raises(
        RuntimeError,
        match="programming bug",
    ):
        service.deliver(
            _source_explanation()
        )


def test_incorrect_source_disclaimer_is_blocked() -> None:
    source = _source_explanation(
        disclaimer=(
            "This explanation is evidence."
        )
    )

    service = (
        ModelExplanationDeliveryService(
            generator=StaticGenerator(
                result=_live_generation()
            )
        )
    )

    with pytest.raises(
        ValueError,
        match="incorrect disclaimer",
    ):
        service.deliver(
            source
        )


def test_missing_disclaimer_in_source_text_is_blocked() -> None:
    source = _source_explanation(
        explanation_text=(
            "Risk-increasing factors were found."
        )
    )

    service = (
        ModelExplanationDeliveryService(
            generator=StaticGenerator(
                result=_live_generation()
            )
        )
    )

    with pytest.raises(
        ValueError,
        match="missing the required disclaimer",
    ):
        service.deliver(
            source
        )


def test_convenience_function_uses_service() -> None:
    source = _source_explanation()

    service = (
        ModelExplanationDeliveryService(
            generator=StaticGenerator(
                result=_live_generation()
            )
        )
    )

    result = deliver_model_explanation(
        source,
        service=service,
    )

    assert result.delivery_mode == "LIVE_OPENAI"
    assert result.payment_id == "pay_test_001"
    assert result.label == EXPECTED_LABEL