from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    timezone,
)
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.support_signal_extractor import (
    OPENAI_EXTRACTION_VERSION,
    SupportSignalGenerationDisabledError,
    SupportSignalGenerationResult,
)
from src.support_signal_service import (
    FileSupportSignalCache,
    SupportSignalService,
    SupportSignalServiceConfig,
)
from src.support_text import (
    EVENT_COLUMNS,
    SupportSignals,
    compile_support_text,
)


NOW = datetime(
    2026,
    1,
    1,
    tzinfo=timezone.utc,
)

SCORING_AT = datetime(
    2025,
    6,
    8,
    tzinfo=timezone.utc,
)


class StaticExtractor:
    def __init__(
        self,
        *,
        result: (
            SupportSignalGenerationResult
            | None
        ) = None,
        error: BaseException | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[
            Any
        ] = []

        self.config = (
            SimpleNamespace(
                model="gpt-test"
            )
        )

    def generate(
        self,
        compiled: Any,
    ) -> SupportSignalGenerationResult:
        self.calls.append(
            compiled
        )

        if self.error is not None:
            raise self.error

        assert self.result is not None
        return self.result


def _events(
    *,
    payment_id: str,
    text: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "support_event_id": (
                    "sup_test_001"
                ),
                "payment_id": payment_id,
                "event_at": datetime(
                    2025,
                    6,
                    5,
                    tzinfo=timezone.utc,
                ),
                "channel": "chat",
                "message_text": text,
            }
        ],
        columns=EVENT_COLUMNS,
    )


def _compiled(
    *,
    payment_id: str = "pay_test_001",
    text: str = (
        "The order has not arrived."
    ),
):
    return compile_support_text(
        payment_id=payment_id,
        scoring_at=SCORING_AT,
        events=_events(
            payment_id=payment_id,
            text=text,
        ),
    )


def _empty_compiled(
    *,
    payment_id: str = "pay_empty",
):
    return compile_support_text(
        payment_id=payment_id,
        scoring_at=SCORING_AT,
        events=pd.DataFrame(
            columns=EVENT_COLUMNS
        ),
    )


def _openai_result(
    compiled: Any,
) -> SupportSignalGenerationResult:
    return SupportSignalGenerationResult(
        signals=SupportSignals(
            intent_to_cancel=False,
            non_receipt_complaint=True,
            dissatisfaction=False,
        ),
        provider="openai",
        model="gpt-test",
        response_id="resp_test_001",
        latency_ms=12,
        input_hash=(
            compiled.input_hash
        ),
        extraction_version=(
            OPENAI_EXTRACTION_VERSION
        ),
    )


def _cache(
    tmp_path: Path,
    *,
    now: datetime = NOW,
) -> FileSupportSignalCache:
    return FileSupportSignalCache(
        tmp_path
        / "cache",
        now_factory=lambda: now,
    )


def _config() -> SupportSignalServiceConfig:
    return SupportSignalServiceConfig(
        generation_version=(
            "service-test-v1"
        ),
        cache_ttl_seconds=3_600,
        fallback_cache_ttl_seconds=60,
    )


def test_live_result_is_cached(
    tmp_path: Path,
) -> None:
    compiled = _compiled()

    extractor = StaticExtractor(
        result=_openai_result(
            compiled
        )
    )

    service = SupportSignalService(
        extractor=extractor,
        cache=_cache(
            tmp_path
        ),
        config=_config(),
    )

    first = service.generate(
        compiled
    )

    second = service.generate(
        compiled
    )

    assert (
        first.delivery_mode
        == "LIVE_OPENAI"
    )
    assert first.cache_hit is False
    assert first.fallback_used is False
    assert first.provider == "openai"
    assert first.latency_ms == 12

    assert second.delivery_mode == "CACHE"
    assert second.cache_hit is True
    assert second.fallback_used is False
    assert second.provider == "openai"
    assert second.latency_ms == 0

    assert first.signals == second.signals
    assert len(extractor.calls) == 1


def test_equivalent_text_reuses_hash_cache(
    tmp_path: Path,
) -> None:
    first_compiled = _compiled(
        payment_id="pay_first"
    )

    second_compiled = _compiled(
        payment_id="pay_second"
    )

    assert (
        first_compiled.input_hash
        == second_compiled.input_hash
    )

    extractor = StaticExtractor(
        result=_openai_result(
            first_compiled
        )
    )

    service = SupportSignalService(
        extractor=extractor,
        cache=_cache(
            tmp_path
        ),
        config=_config(),
    )

    first = service.generate(
        first_compiled
    )

    second = service.generate(
        second_compiled
    )

    assert (
        first.delivery_mode
        == "LIVE_OPENAI"
    )
    assert second.delivery_mode == "CACHE"

    assert (
        second.payment_id
        == "pay_second"
    )

    assert len(extractor.calls) == 1


def test_refresh_bypasses_cache(
    tmp_path: Path,
) -> None:
    compiled = _compiled()

    extractor = StaticExtractor(
        result=_openai_result(
            compiled
        )
    )

    service = SupportSignalService(
        extractor=extractor,
        cache=_cache(
            tmp_path
        ),
        config=_config(),
    )

    service.generate(
        compiled
    )

    refreshed = service.generate(
        compiled,
        refresh=True,
    )

    assert (
        refreshed.delivery_mode
        == "LIVE_OPENAI"
    )
    assert refreshed.cache_hit is False
    assert len(extractor.calls) == 2


def test_disabled_ai_uses_deterministic_fallback(
    tmp_path: Path,
) -> None:
    compiled = _compiled(
        text="Please cancel this order."
    )

    service = SupportSignalService(
        extractor=StaticExtractor(
            error=(
                SupportSignalGenerationDisabledError(
                    "disabled"
                )
            )
        ),
        cache=_cache(
            tmp_path
        ),
        config=_config(),
    )

    result = service.generate(
        compiled
    )

    assert (
        result.delivery_mode
        == "DETERMINISTIC_FALLBACK"
    )
    assert (
        result.provider
        == "deterministic_fallback"
    )
    assert result.cache_hit is False
    assert result.fallback_used is True
    assert (
        result.fallback_reason
        == "AI_DISABLED"
    )
    assert (
        result.signals.intent_to_cancel
        is True
    )


def test_fallback_is_cached(
    tmp_path: Path,
) -> None:
    compiled = _compiled(
        text="Please cancel this order."
    )

    extractor = StaticExtractor(
        error=(
            SupportSignalGenerationDisabledError(
                "disabled"
            )
        )
    )

    service = SupportSignalService(
        extractor=extractor,
        cache=_cache(
            tmp_path
        ),
        config=_config(),
    )

    first = service.generate(
        compiled
    )

    second = service.generate(
        compiled
    )

    assert (
        first.delivery_mode
        == "DETERMINISTIC_FALLBACK"
    )
    assert second.delivery_mode == "CACHE"
    assert second.cache_hit is True
    assert second.fallback_used is True
    assert (
        second.fallback_reason
        == "CACHED_FALLBACK"
    )
    assert len(extractor.calls) == 1


def test_no_text_skips_openai(
    tmp_path: Path,
) -> None:
    compiled = _empty_compiled()

    extractor = StaticExtractor(
        error=AssertionError(
            "must not be called"
        )
    )

    service = SupportSignalService(
        extractor=extractor,
        cache=_cache(
            tmp_path
        ),
        config=_config(),
    )

    result = service.generate(
        compiled
    )

    assert (
        result.delivery_mode
        == "DETERMINISTIC_FALLBACK"
    )
    assert (
        result.fallback_reason
        == "NO_OBSERVABLE_TEXT"
    )
    assert not any(
        result.signals
        .model_dump()
        .values()
    )
    assert extractor.calls == []


def test_programming_error_is_not_hidden(
    tmp_path: Path,
) -> None:
    service = SupportSignalService(
        extractor=StaticExtractor(
            error=RuntimeError(
                "programming bug"
            )
        ),
        cache=_cache(
            tmp_path
        ),
        config=_config(),
    )

    with pytest.raises(
        RuntimeError,
        match="programming bug",
    ):
        service.generate(
            _compiled()
        )


def test_corrupt_cache_recovers_safely(
    tmp_path: Path,
) -> None:
    compiled = _compiled()

    cache = _cache(
        tmp_path
    )

    extractor = StaticExtractor(
        result=_openai_result(
            compiled
        )
    )

    service = SupportSignalService(
        extractor=extractor,
        cache=cache,
        config=_config(),
    )

    path = cache._cache_path(
        input_hash=(
            compiled.input_hash
        ),
        generation_version=(
            service.cache_generation_version
        ),
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "{not valid json",
        encoding="utf-8",
    )

    result = service.generate(
        compiled
    )

    assert (
        result.delivery_mode
        == "LIVE_OPENAI"
    )
    assert len(extractor.calls) == 1


def test_expired_cache_is_ignored(
    tmp_path: Path,
) -> None:
    compiled = _compiled()

    write_cache = _cache(
        tmp_path,
        now=NOW,
    )

    generation = _openai_result(
        compiled
    )

    generation_version = (
        "service-test-v1:gpt-test"
    )

    write_cache.put(
        generation=generation,
        generation_version=(
            generation_version
        ),
        ttl_seconds=10,
    )

    expired_cache = _cache(
        tmp_path,
        now=(
            NOW
            + timedelta(
                seconds=11
            )
        ),
    )

    assert (
        expired_cache.get(
            input_hash=(
                compiled.input_hash
            ),
            generation_version=(
                generation_version
            ),
        )
        is None
    )


def test_cache_contains_no_payment_identity(
    tmp_path: Path,
) -> None:
    compiled = _compiled()

    cache = _cache(
        tmp_path
    )

    service = SupportSignalService(
        extractor=StaticExtractor(
            result=_openai_result(
                compiled
            )
        ),
        cache=cache,
        config=_config(),
    )

    service.generate(
        compiled
    )

    cache_files = list(
        (
            tmp_path
            / "cache"
        ).glob(
            "*.json"
        )
    )

    assert len(cache_files) == 1

    content = cache_files[
        0
    ].read_text(
        encoding="utf-8"
    )

    assert "payment_id" not in content
    assert "pay_test_001" not in content
    assert "customer_id" not in content
    assert "dispute_status" not in content
    assert "reason_code" not in content
    assert (
        "chargeback_within_120d"
        not in content
    )


def test_mismatched_extractor_hash_falls_back(
    tmp_path: Path,
) -> None:
    compiled = _compiled(
        text="Please cancel this order."
    )

    mismatched = _openai_result(
        compiled
    ).model_copy(
        update={
            "input_hash": "0" * 64,
        }
    )

    service = SupportSignalService(
        extractor=StaticExtractor(
            result=mismatched
        ),
        cache=_cache(
            tmp_path
        ),
        config=_config(),
    )

    result = service.generate(
        compiled
    )

    assert result.fallback_used is True
    assert (
        result.fallback_reason
        == "OUTPUT_BOUNDARY_BLOCK"
    )
    assert (
        result.signals.intent_to_cancel
        is True
    )