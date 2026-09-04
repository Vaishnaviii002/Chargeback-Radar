from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from src.explanation_text import DISCLAIMER
from src.model_explanation_generator import (
    EXPECTED_LABEL,
    ModelExplanationGenerationDisabledError,
    ModelExplanationGenerationError,
    ModelExplanationGenerationIncompleteError,
    ModelExplanationGenerationRefusedError,
    ModelExplanationGenerationResult,
    ModelExplanationGenerationUnsafeError,
    OpenAIModelExplanationGenerator,
    prepare_factors,
)
from src.model_explanation_service import (
    ExplanationFactor,
    ModelExplanation,
)


OPENAI_PHRASING_VERSION = (
    "openai-phrasing-v1"
)


RECOVERABLE_PHRASING_ERRORS = (
    ModelExplanationGenerationError,
    ValidationError,
)


class ModelExplanationDeliveryResult(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    payment_id: str = Field(
        min_length=1,
    )

    model_version: str = Field(
        min_length=1,
    )

    shap_explanation_version: str = Field(
        min_length=1,
    )

    deterministic_text_version: str = Field(
        min_length=1,
    )

    delivery_text_version: str = Field(
        min_length=1,
    )

    delivery_mode: Literal[
        "LIVE_OPENAI",
        "DETERMINISTIC_FALLBACK",
    ]

    provider: Literal[
        "openai",
        "deterministic_formatter",
    ]

    model: str | None = Field(
        default=None,
        max_length=100,
    )

    response_id: str | None = Field(
        default=None,
        max_length=200,
    )

    latency_ms: int = Field(
        ge=0,
    )

    fallback_used: bool

    fallback_reason: str | None = Field(
        default=None,
        pattern=r"^[A-Z][A-Z0-9_]{2,63}$",
    )

    label: Literal[
        "Model explanation — not evidence"
    ]

    explanation: str = Field(
        min_length=1,
    )

    disclaimer: str = Field(
        min_length=1,
    )

    positive_factors: list[
        ExplanationFactor
    ]

    negative_factors: list[
        ExplanationFactor
    ]


def _exception_chain(
    error: BaseException,
):
    current: BaseException | None = error
    visited: set[int] = set()

    while (
        current is not None
        and id(current) not in visited
    ):
        visited.add(id(current))
        yield current

        current = (
            current.__cause__
            or current.__context__
        )


def _request_id_from_error(
    error: BaseException,
) -> str | None:
    for current in _exception_chain(
        error
    ):
        request_id = getattr(
            current,
            "request_id",
            None,
        )

        if request_id:
            return str(request_id)[:200]

    return None


def _safe_error_category(
    error: BaseException,
) -> str:
    if isinstance(
        error,
        ModelExplanationGenerationDisabledError,
    ):
        return "AI_DISABLED"

    if isinstance(
        error,
        ModelExplanationGenerationRefusedError,
    ):
        return "MODEL_REFUSAL"

    if isinstance(
        error,
        ModelExplanationGenerationIncompleteError,
    ):
        return "INCOMPLETE_RESPONSE"

    if isinstance(
        error,
        ModelExplanationGenerationUnsafeError,
    ):
        return "OUTPUT_GUARDRAIL_BLOCK"

    if isinstance(
        error,
        ValidationError,
    ):
        return "OUTPUT_SCHEMA_INVALID"

    class_names = {
        item.__class__.__name__
        for item in _exception_chain(
            error
        )
    }

    if "APITimeoutError" in class_names:
        return "API_TIMEOUT"

    if "APIConnectionError" in class_names:
        return "API_CONNECTION_FAILED"

    if "RateLimitError" in class_names:
        return "API_RATE_LIMIT"

    if "AuthenticationError" in class_names:
        return "API_AUTHENTICATION_FAILED"

    if "PermissionDeniedError" in class_names:
        return "API_PERMISSION_DENIED"

    if "BadRequestError" in class_names:
        return "API_BAD_REQUEST"

    if "InternalServerError" in class_names:
        return "API_SERVER_ERROR"

    return "API_GENERATION_FAILED"


def _validate_source_explanation(
    explanation: ModelExplanation,
) -> None:
    if explanation.label != EXPECTED_LABEL:
        raise ValueError(
            "Source explanation has an "
            "incorrect safety label"
        )

    if explanation.disclaimer != DISCLAIMER:
        raise ValueError(
            "Source explanation has an "
            "incorrect disclaimer"
        )

    if DISCLAIMER not in explanation.explanation:
        raise ValueError(
            "Source explanation text is missing "
            "the required disclaimer"
        )

    prepare_factors(
        explanation.positive_factors,
        explanation.negative_factors,
    )


def _fallback_result(
    explanation: ModelExplanation,
    *,
    error: BaseException,
) -> ModelExplanationDeliveryResult:
    return ModelExplanationDeliveryResult(
        payment_id=explanation.payment_id,
        model_version=explanation.model_version,
        shap_explanation_version=(
            explanation.shap_explanation_version
        ),
        deterministic_text_version=(
            explanation.text_explanation_version
        ),
        delivery_text_version=(
            explanation.text_explanation_version
        ),
        delivery_mode=(
            "DETERMINISTIC_FALLBACK"
        ),
        provider="deterministic_formatter",
        model=None,
        response_id=(
            _request_id_from_error(
                error
            )
        ),
        latency_ms=0,
        fallback_used=True,
        fallback_reason=(
            _safe_error_category(
                error
            )
        ),
        label=explanation.label,
        explanation=explanation.explanation,
        disclaimer=explanation.disclaimer,
        positive_factors=(
            explanation.positive_factors
        ),
        negative_factors=(
            explanation.negative_factors
        ),
    )


def _live_result(
    explanation: ModelExplanation,
    generation: (
        ModelExplanationGenerationResult
    ),
) -> ModelExplanationDeliveryResult:
    summary = (
        generation.draft.summary.strip()
    )

    final_text = (
        summary
        + "\n\n"
        + explanation.explanation
    )

    if DISCLAIMER not in final_text:
        raise AssertionError(
            "Delivered explanation is missing "
            "the required disclaimer"
        )

    return ModelExplanationDeliveryResult(
        payment_id=explanation.payment_id,
        model_version=explanation.model_version,
        shap_explanation_version=(
            explanation.shap_explanation_version
        ),
        deterministic_text_version=(
            explanation.text_explanation_version
        ),
        delivery_text_version=(
            OPENAI_PHRASING_VERSION
        ),
        delivery_mode="LIVE_OPENAI",
        provider="openai",
        model=generation.model,
        response_id=generation.response_id,
        latency_ms=generation.latency_ms,
        fallback_used=False,
        fallback_reason=None,
        label=explanation.label,
        explanation=final_text,
        disclaimer=explanation.disclaimer,
        positive_factors=(
            explanation.positive_factors
        ),
        negative_factors=(
            explanation.negative_factors
        ),
    )


class ModelExplanationDeliveryService:
    """
    Deliver optional OpenAI phrasing with a deterministic fallback.

    The validated deterministic explanation remains the authoritative
    base. OpenAI cannot change risk, factors, values or directions.
    """

    def __init__(
        self,
        *,
        generator: Any | None = None,
    ) -> None:
        self.generator = (
            generator
            or OpenAIModelExplanationGenerator()
        )

    def deliver(
        self,
        explanation: ModelExplanation,
    ) -> ModelExplanationDeliveryResult:
        _validate_source_explanation(
            explanation
        )

        try:
            generation = (
                self.generator.generate(
                    positive_factors=(
                        explanation.positive_factors
                    ),
                    negative_factors=(
                        explanation.negative_factors
                    ),
                )
            )
        except (
            RECOVERABLE_PHRASING_ERRORS
        ) as error:
            return _fallback_result(
                explanation,
                error=error,
            )

        return _live_result(
            explanation,
            generation,
        )


def deliver_model_explanation(
    explanation: ModelExplanation,
    *,
    service: (
        ModelExplanationDeliveryService
        | None
    ) = None,
) -> ModelExplanationDeliveryResult:
    return (
        service
        or ModelExplanationDeliveryService()
    ).deliver(
        explanation
    )