from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.evidence_contract import (
    EvidenceCase,
    EvidenceContractError,
    EvidencePack,
)
from src.evidence_fallback import generate_fallback_evidence_pack
from src.evidence_generator import (
    EvidenceGenerationDisabledError,
    EvidenceGenerationError,
    EvidenceGenerationIncompleteError,
    EvidenceGenerationRefusedError,
    OpenAIEvidenceGenerator,
)
from src.evidence_guardrails import (
    EvidenceGuardrailError,
    GuardrailReport,
    validate_evidence_case_safety,
    validate_generated_evidence_pack,
)
from src.evidence_store import (
    EvidenceAuditLog,
    EvidenceStoreError,
    FileEvidenceCache,
)


RECOVERABLE_GENERATION_ERRORS = (
    EvidenceGenerationError,
    EvidenceContractError,
    EvidenceGuardrailError,
    ValidationError,
)


@dataclass(frozen=True)
class EvidenceServiceConfig:
    generation_version: str = "evidence-v1"
    fallback_cache_ttl_seconds: int = 300

    @classmethod
    def from_env(cls) -> "EvidenceServiceConfig":
        try:
            from dotenv import load_dotenv

            load_dotenv(override=False)
        except ImportError:
            pass

        return cls(
            generation_version=os.getenv(
                "EVIDENCE_GENERATION_VERSION",
                "evidence-v1",
            ),
            fallback_cache_ttl_seconds=int(
                os.getenv(
                    "EVIDENCE_FALLBACK_CACHE_TTL_SECONDS",
                    "300",
                )
            ),
        )

    def __post_init__(self) -> None:
        if not self.generation_version.strip():
            raise ValueError("generation_version must not be empty")
        if self.fallback_cache_ttl_seconds < 1:
            raise ValueError(
                "fallback_cache_ttl_seconds must be positive"
            )


class EvidenceDeliveryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_pack: EvidencePack
    delivery_mode: Literal[
        "LIVE_OPENAI",
        "CACHE",
        "DETERMINISTIC_FALLBACK",
    ]
    provider: str = Field(min_length=1, max_length=80)
    model: str | None = Field(default=None, max_length=100)
    response_id: str | None = Field(default=None, max_length=200)
    latency_ms: int = Field(ge=0)
    cache_hit: bool
    fallback_used: bool
    fallback_reason: str | None = Field(
        default=None,
        pattern=r"^[A-Z][A-Z0-9_]{2,63}$",
    )
    guardrails: GuardrailReport


def _exception_chain(error: BaseException):
    current: BaseException | None = error
    visited: set[int] = set()

    while current is not None and id(current) not in visited:
        visited.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _request_id_from_error(error: BaseException) -> str | None:
    for current in _exception_chain(error):
        request_id = getattr(current, "request_id", None)
        if request_id:
            return str(request_id)[:200]
    return None


def _safe_error_category(error: BaseException) -> str:
    if isinstance(error, EvidenceGenerationDisabledError):
        return "AI_DISABLED"
    if isinstance(error, EvidenceGenerationRefusedError):
        return "MODEL_REFUSAL"
    if isinstance(error, EvidenceGenerationIncompleteError):
        return "INCOMPLETE_RESPONSE"
    if isinstance(error, EvidenceContractError):
        return "OUTPUT_CONTRACT_VIOLATION"
    if isinstance(error, EvidenceGuardrailError):
        return "OUTPUT_GUARDRAIL_BLOCK"
    if isinstance(error, ValidationError):
        return "OUTPUT_SCHEMA_INVALID"

    class_names = {
        item.__class__.__name__
        for item in _exception_chain(error)
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


class EvidenceService:
    """Resilient evidence delivery with cache, fallback, and audit."""

    def __init__(
        self,
        *,
        generator: Any | None = None,
        cache: FileEvidenceCache | None = None,
        audit: EvidenceAuditLog | None = None,
        config: EvidenceServiceConfig | None = None,
    ) -> None:
        self.generator = generator or OpenAIEvidenceGenerator()
        self.cache = cache or FileEvidenceCache.from_env()
        self.audit = audit or EvidenceAuditLog.from_env()
        self.config = config or EvidenceServiceConfig.from_env()

    @property
    def cache_generation_version(self) -> str:
        generator_config = getattr(self.generator, "config", None)
        model = getattr(generator_config, "model", "unknown-model")
        return f"{self.config.generation_version}:{model}"

    def _record_cache_failure(
        self,
        *,
        event_type: Literal["CACHE_READ_FAILED", "CACHE_WRITE_FAILED"],
        evidence_case: EvidenceCase,
        evidence_pack: EvidencePack | None = None,
    ) -> None:
        self.audit.append(
            event_type=event_type,
            status="FAILURE",
            evidence_case=evidence_case,
            evidence_pack=evidence_pack,
            provider="cache",
            error_category=event_type,
        )

    def _cache_result(
        self,
        evidence_case: EvidenceCase,
    ) -> EvidenceDeliveryResult | None:
        try:
            entry = self.cache.get(
                evidence_case,
                generation_version=self.cache_generation_version,
            )
        except (
            EvidenceStoreError,
            EvidenceContractError,
            EvidenceGuardrailError,
            ValidationError,
            OSError,
        ):
            self._record_cache_failure(
                event_type="CACHE_READ_FAILED",
                evidence_case=evidence_case,
            )
            return None

        if entry is None:
            return None

        guardrails = validate_generated_evidence_pack(
            entry.evidence_pack,
            evidence_case,
        )
        self.audit.append(
            event_type="CACHE_HIT",
            status="SUCCESS",
            evidence_case=evidence_case,
            evidence_pack=entry.evidence_pack,
            provider=entry.provider,
            model=entry.model,
            response_id=entry.response_id,
            cache_hit=True,
            latency_ms=0,
        )

        fallback_cached = entry.provider == "deterministic_fallback"
        return EvidenceDeliveryResult(
            evidence_pack=entry.evidence_pack,
            delivery_mode="CACHE",
            provider=entry.provider,
            model=entry.model,
            response_id=entry.response_id,
            latency_ms=0,
            cache_hit=True,
            fallback_used=fallback_cached,
            fallback_reason=(
                "CACHED_FALLBACK"
                if fallback_cached
                else None
            ),
            guardrails=guardrails,
        )

    def _write_cache(
        self,
        *,
        evidence_case: EvidenceCase,
        evidence_pack: EvidencePack,
        provider: str,
        model: str | None,
        response_id: str | None,
        ttl_seconds: int | None = None,
    ) -> None:
        try:
            self.cache.put(
                evidence_case,
                evidence_pack,
                provider=provider,
                model=model,
                response_id=response_id,
                generation_version=self.cache_generation_version,
                ttl_seconds=ttl_seconds,
            )
        except (
            EvidenceStoreError,
            OSError,
            ValueError,
        ):
            # Evidence remains usable; the cache failure is auditable.
            self._record_cache_failure(
                event_type="CACHE_WRITE_FAILED",
                evidence_case=evidence_case,
                evidence_pack=evidence_pack,
            )

    def generate(
        self,
        evidence_case: EvidenceCase,
        *,
        refresh: bool = False,
    ) -> EvidenceDeliveryResult:
        # Establish a safe input boundary before cache or fallback handling.
        validate_evidence_case_safety(evidence_case)

        if not refresh:
            cached = self._cache_result(evidence_case)
            if cached is not None:
                return cached

        try:
            live_result = self.generator.generate(evidence_case)
        except RECOVERABLE_GENERATION_ERRORS as error:
            category = _safe_error_category(error)
            failed_request_id = _request_id_from_error(error)
            self.audit.append(
                event_type="GENERATION_FAILED",
                status="FAILURE",
                evidence_case=evidence_case,
                provider="openai",
                model=getattr(
                    getattr(self.generator, "config", None),
                    "model",
                    None,
                ),
                response_id=failed_request_id,
                error_category=category,
            )

            fallback_pack = generate_fallback_evidence_pack(evidence_case)
            fallback_guardrails = validate_generated_evidence_pack(
                fallback_pack,
                evidence_case,
            )
            self.audit.append(
                event_type="FALLBACK_USED",
                status="SUCCESS",
                evidence_case=evidence_case,
                evidence_pack=fallback_pack,
                provider="deterministic_fallback",
                latency_ms=0,
            )
            self._write_cache(
                evidence_case=evidence_case,
                evidence_pack=fallback_pack,
                provider="deterministic_fallback",
                model=None,
                response_id=None,
                ttl_seconds=(
                    self.config.fallback_cache_ttl_seconds
                ),
            )

            return EvidenceDeliveryResult(
                evidence_pack=fallback_pack,
                delivery_mode="DETERMINISTIC_FALLBACK",
                provider="deterministic_fallback",
                model=None,
                response_id=failed_request_id,
                latency_ms=0,
                cache_hit=False,
                fallback_used=True,
                fallback_reason=category,
                guardrails=fallback_guardrails,
            )

        self.audit.append(
            event_type="EVIDENCE_GENERATED",
            status="SUCCESS",
            evidence_case=evidence_case,
            evidence_pack=live_result.evidence_pack,
            provider="openai",
            model=live_result.model,
            response_id=live_result.response_id,
            latency_ms=live_result.latency_ms,
        )
        self._write_cache(
            evidence_case=evidence_case,
            evidence_pack=live_result.evidence_pack,
            provider="openai",
            model=live_result.model,
            response_id=live_result.response_id,
        )

        return EvidenceDeliveryResult(
            evidence_pack=live_result.evidence_pack,
            delivery_mode="LIVE_OPENAI",
            provider="openai",
            model=live_result.model,
            response_id=live_result.response_id,
            latency_ms=live_result.latency_ms,
            cache_hit=False,
            fallback_used=False,
            fallback_reason=None,
            guardrails=live_result.guardrails,
        )


def generate_evidence_resiliently(
    evidence_case: EvidenceCase,
    *,
    refresh: bool = False,
    service: EvidenceService | None = None,
) -> EvidenceDeliveryResult:
    return (service or EvidenceService()).generate(
        evidence_case,
        refresh=refresh,
    )
