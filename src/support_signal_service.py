from __future__ import annotations

from dataclasses import dataclass
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import (
    Any,
    Callable,
    Literal,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from src.support_signal_extractor import (
    OpenAISupportSignalExtractor,
    SupportSignalBoundaryError,
    SupportSignalGenerationDisabledError,
    SupportSignalGenerationError,
    SupportSignalGenerationIncompleteError,
    SupportSignalGenerationRefusedError,
    SupportSignalGenerationResult,
    deterministic_extract_support_signals,
    validate_model_boundary,
)
from src.support_text import (
    CompiledSupportText,
    SupportSignals,
)


ROOT = Path(__file__).resolve().parents[1]

CACHE_SCHEMA_VERSION = (
    "support-signal-cache-v1"
)


RECOVERABLE_EXTRACTION_ERRORS = (
    SupportSignalGenerationError,
    ValidationError,
)


class SupportSignalCacheError(
    RuntimeError
):
    pass


@dataclass(frozen=True)
class SupportSignalServiceConfig:
    generation_version: str = (
        "support-signal-service-v1"
    )

    cache_ttl_seconds: int = 86_400

    fallback_cache_ttl_seconds: int = 300

    @classmethod
    def from_env(
        cls,
    ) -> "SupportSignalServiceConfig":
        try:
            from dotenv import load_dotenv

            load_dotenv(
                override=False
            )
        except ImportError:
            pass

        return cls(
            generation_version=os.getenv(
                "SUPPORT_SIGNAL_GENERATION_VERSION",
                "support-signal-service-v1",
            ),
            cache_ttl_seconds=int(
                os.getenv(
                    "SUPPORT_SIGNAL_CACHE_TTL_SECONDS",
                    "86400",
                )
            ),
            fallback_cache_ttl_seconds=int(
                os.getenv(
                    "SUPPORT_SIGNAL_FALLBACK_CACHE_TTL_SECONDS",
                    "300",
                )
            ),
        )

    def __post_init__(self) -> None:
        if not self.generation_version.strip():
            raise ValueError(
                "generation_version must not "
                "be empty"
            )

        if self.cache_ttl_seconds < 1:
            raise ValueError(
                "cache_ttl_seconds must be "
                "positive"
            )

        if (
            self.fallback_cache_ttl_seconds
            < 1
        ):
            raise ValueError(
                "fallback_cache_ttl_seconds "
                "must be positive"
            )


class SupportSignalCacheEntry(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    cache_schema_version: Literal[
        "support-signal-cache-v1"
    ]

    input_hash: str = Field(
        pattern=r"^[a-f0-9]{64}$",
    )

    generation_version: str = Field(
        min_length=1,
        max_length=200,
    )

    signals: SupportSignals

    provider: Literal[
        "openai",
        "deterministic_fallback",
    ]

    model: str | None = Field(
        default=None,
        max_length=100,
    )

    response_id: str | None = Field(
        default=None,
        max_length=200,
    )

    extraction_version: str = Field(
        min_length=1,
        max_length=100,
    )

    created_at: datetime

    expires_at: datetime

    @field_validator(
        "created_at",
        "expires_at",
    )
    @classmethod
    def validate_timestamp(
        cls,
        value: datetime,
    ) -> datetime:
        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "Cache timestamps must be "
                "timezone-aware"
            )

        return value


class SupportSignalDeliveryResult(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    payment_id: str = Field(
        min_length=1,
    )

    signals: SupportSignals

    delivery_mode: Literal[
        "LIVE_OPENAI",
        "CACHE",
        "DETERMINISTIC_FALLBACK",
    ]

    provider: Literal[
        "openai",
        "deterministic_fallback",
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

    input_hash: str = Field(
        pattern=r"^[a-f0-9]{64}$",
    )

    extraction_version: str = Field(
        min_length=1,
        max_length=100,
    )

    cache_hit: bool

    fallback_used: bool

    fallback_reason: str | None = Field(
        default=None,
        pattern=r"^[A-Z][A-Z0-9_]{2,63}$",
    )


def _utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


class FileSupportSignalCache:
    def __init__(
        self,
        cache_dir: Path,
        *,
        now_factory: Callable[
            [],
            datetime,
        ] = _utc_now,
    ) -> None:
        self.cache_dir = Path(
            cache_dir
        )

        self.now_factory = (
            now_factory
        )

    @classmethod
    def from_env(
        cls,
    ) -> "FileSupportSignalCache":
        try:
            from dotenv import load_dotenv

            load_dotenv(
                override=False
            )
        except ImportError:
            pass

        configured = os.getenv(
            "SUPPORT_SIGNAL_CACHE_DIR",
            "runtime/support_signal_cache",
        )

        cache_dir = Path(
            configured
        )

        if not cache_dir.is_absolute():
            cache_dir = (
                ROOT
                / cache_dir
            )

        return cls(
            cache_dir
        )

    def _cache_path(
        self,
        *,
        input_hash: str,
        generation_version: str,
    ) -> Path:
        key_material = (
            generation_version
            + "|"
            + input_hash
        )

        key = sha256(
            key_material.encode(
                "utf-8"
            )
        ).hexdigest()

        return (
            self.cache_dir
            / f"{key}.json"
        )

    def get(
        self,
        *,
        input_hash: str,
        generation_version: str,
    ) -> (
        SupportSignalCacheEntry
        | None
    ):
        path = self._cache_path(
            input_hash=input_hash,
            generation_version=(
                generation_version
            ),
        )

        if not path.exists():
            return None

        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            entry = (
                SupportSignalCacheEntry
                .model_validate(
                    payload
                )
            )
        except (
            OSError,
            json.JSONDecodeError,
            ValidationError,
        ) as error:
            raise SupportSignalCacheError(
                "Support-signal cache entry "
                "is unreadable or invalid"
            ) from error

        if entry.input_hash != input_hash:
            raise SupportSignalCacheError(
                "Support-signal cache input "
                "hash mismatch"
            )

        if (
            entry.generation_version
            != generation_version
        ):
            raise SupportSignalCacheError(
                "Support-signal cache "
                "generation-version mismatch"
            )

        now = self.now_factory()

        if (
            now.tzinfo is None
            or now.utcoffset() is None
        ):
            raise SupportSignalCacheError(
                "Cache clock must be "
                "timezone-aware"
            )

        if entry.expires_at <= now:
            return None

        return entry

    def put(
        self,
        *,
        generation: (
            SupportSignalGenerationResult
        ),
        generation_version: str,
        ttl_seconds: int,
    ) -> None:
        if ttl_seconds < 1:
            raise ValueError(
                "ttl_seconds must be positive"
            )

        now = self.now_factory()

        if (
            now.tzinfo is None
            or now.utcoffset() is None
        ):
            raise SupportSignalCacheError(
                "Cache clock must be "
                "timezone-aware"
            )

        entry = SupportSignalCacheEntry(
            cache_schema_version=(
                CACHE_SCHEMA_VERSION
            ),
            input_hash=(
                generation.input_hash
            ),
            generation_version=(
                generation_version
            ),
            signals=generation.signals,
            provider=generation.provider,
            model=generation.model,
            response_id=(
                generation.response_id
            ),
            extraction_version=(
                generation.extraction_version
            ),
            created_at=now,
            expires_at=(
                now
                + timedelta(
                    seconds=ttl_seconds
                )
            ),
        )

        path = self._cache_path(
            input_hash=(
                generation.input_hash
            ),
            generation_version=(
                generation_version
            ),
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = (
            path.with_suffix(
                ".json.tmp"
            )
        )

        try:
            temporary_path.write_text(
                entry.model_dump_json(
                    indent=2
                ),
                encoding="utf-8",
            )

            temporary_path.replace(
                path
            )
        except OSError as error:
            raise SupportSignalCacheError(
                "Unable to write support-"
                "signal cache entry"
            ) from error


def _exception_chain(
    error: BaseException,
):
    current: BaseException | None = error
    visited: set[int] = set()

    while (
        current is not None
        and id(current) not in visited
    ):
        visited.add(
            id(current)
        )

        yield current

        current = (
            current.__cause__
            or current.__context__
        )


def _safe_error_category(
    error: BaseException,
) -> str:
    if isinstance(
        error,
        SupportSignalGenerationDisabledError,
    ):
        return "AI_DISABLED"

    if isinstance(
        error,
        SupportSignalGenerationRefusedError,
    ):
        return "MODEL_REFUSAL"

    if isinstance(
        error,
        SupportSignalGenerationIncompleteError,
    ):
        return "INCOMPLETE_RESPONSE"

    if isinstance(
        error,
        SupportSignalBoundaryError,
    ):
        return "OUTPUT_BOUNDARY_BLOCK"

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


class SupportSignalService:
    """
    Input-hash cache plus bounded OpenAI extraction and
    deterministic recovery.
    """

    def __init__(
        self,
        *,
        extractor: Any | None = None,
        cache: (
            FileSupportSignalCache
            | None
        ) = None,
        config: (
            SupportSignalServiceConfig
            | None
        ) = None,
    ) -> None:
        self.extractor = (
            extractor
            or OpenAISupportSignalExtractor()
        )

        self.cache = (
            cache
            or FileSupportSignalCache.from_env()
        )

        self.config = (
            config
            or SupportSignalServiceConfig.from_env()
        )

    @property
    def cache_generation_version(
        self,
    ) -> str:
        extractor_config = getattr(
            self.extractor,
            "config",
            None,
        )

        model = getattr(
            extractor_config,
            "model",
            "unknown-model",
        )

        return (
            self.config.generation_version
            + ":"
            + str(model)
        )

    def _cached_result(
        self,
        compiled: CompiledSupportText,
    ) -> (
        SupportSignalDeliveryResult
        | None
    ):
        try:
            entry = self.cache.get(
                input_hash=(
                    compiled.input_hash
                ),
                generation_version=(
                    self.cache_generation_version
                ),
            )
        except (
            SupportSignalCacheError,
            OSError,
        ):
            return None

        if entry is None:
            return None

        fallback_cached = (
            entry.provider
            == "deterministic_fallback"
        )

        return SupportSignalDeliveryResult(
            payment_id=(
                compiled.payment_id
            ),
            signals=entry.signals,
            delivery_mode="CACHE",
            provider=entry.provider,
            model=entry.model,
            response_id=(
                entry.response_id
            ),
            latency_ms=0,
            input_hash=entry.input_hash,
            extraction_version=(
                entry.extraction_version
            ),
            cache_hit=True,
            fallback_used=(
                fallback_cached
            ),
            fallback_reason=(
                "CACHED_FALLBACK"
                if fallback_cached
                else None
            ),
        )

    def _write_cache(
        self,
        generation: (
            SupportSignalGenerationResult
        ),
        *,
        ttl_seconds: int,
    ) -> None:
        try:
            self.cache.put(
                generation=generation,
                generation_version=(
                    self.cache_generation_version
                ),
                ttl_seconds=ttl_seconds,
            )
        except (
            SupportSignalCacheError,
            OSError,
            ValueError,
        ):
            # A cache failure must not prevent safe signal delivery.
            return

    def _delivery_from_generation(
        self,
        *,
        compiled: CompiledSupportText,
        generation: (
            SupportSignalGenerationResult
        ),
        delivery_mode: Literal[
            "LIVE_OPENAI",
            "DETERMINISTIC_FALLBACK",
        ],
        fallback_reason: str | None,
    ) -> SupportSignalDeliveryResult:
        fallback_used = (
            delivery_mode
            == "DETERMINISTIC_FALLBACK"
        )

        return SupportSignalDeliveryResult(
            payment_id=(
                compiled.payment_id
            ),
            signals=(
                generation.signals
            ),
            delivery_mode=(
                delivery_mode
            ),
            provider=(
                generation.provider
            ),
            model=generation.model,
            response_id=(
                generation.response_id
            ),
            latency_ms=(
                generation.latency_ms
            ),
            input_hash=(
                generation.input_hash
            ),
            extraction_version=(
                generation.extraction_version
            ),
            cache_hit=False,
            fallback_used=(
                fallback_used
            ),
            fallback_reason=(
                fallback_reason
            ),
        )

    def generate(
        self,
        compiled: CompiledSupportText,
        *,
        refresh: bool = False,
    ) -> SupportSignalDeliveryResult:
        validate_model_boundary(
            compiled
        )

        if not refresh:
            cached = self._cached_result(
                compiled
            )

            if cached is not None:
                return cached

        if not compiled.events:
            deterministic = (
                deterministic_extract_support_signals(
                    compiled
                )
            )

            self._write_cache(
                deterministic,
                ttl_seconds=(
                    self.config
                    .fallback_cache_ttl_seconds
                ),
            )

            return self._delivery_from_generation(
                compiled=compiled,
                generation=deterministic,
                delivery_mode=(
                    "DETERMINISTIC_FALLBACK"
                ),
                fallback_reason=(
                    "NO_OBSERVABLE_TEXT"
                ),
            )

        try:
            generation = (
                self.extractor.generate(
                    compiled
                )
            )

            if (
                generation.input_hash
                != compiled.input_hash
            ):
                raise SupportSignalBoundaryError(
                    "Extractor returned a "
                    "different input hash"
                )
        except (
            RECOVERABLE_EXTRACTION_ERRORS
        ) as error:
            deterministic = (
                deterministic_extract_support_signals(
                    compiled
                )
            )

            self._write_cache(
                deterministic,
                ttl_seconds=(
                    self.config
                    .fallback_cache_ttl_seconds
                ),
            )

            return self._delivery_from_generation(
                compiled=compiled,
                generation=deterministic,
                delivery_mode=(
                    "DETERMINISTIC_FALLBACK"
                ),
                fallback_reason=(
                    _safe_error_category(
                        error
                    )
                ),
            )

        self._write_cache(
            generation,
            ttl_seconds=(
                self.config
                .cache_ttl_seconds
            ),
        )

        return self._delivery_from_generation(
            compiled=compiled,
            generation=generation,
            delivery_mode="LIVE_OPENAI",
            fallback_reason=None,
        )


def generate_support_signals_resiliently(
    compiled: CompiledSupportText,
    *,
    refresh: bool = False,
    service: (
        SupportSignalService
        | None
    ) = None,
) -> SupportSignalDeliveryResult:
    return (
        service
        or SupportSignalService()
    ).generate(
        compiled,
        refresh=refresh,
    )