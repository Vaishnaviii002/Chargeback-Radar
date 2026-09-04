from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from src.support_signal_extractor import (
    SupportSignalGenerationDisabledError,
)
from src.support_signal_report import (
    FORBIDDEN_REPORT_COLUMNS,
    SUPPORT_SIGNAL_REPORT_VERSION,
    generate_support_signal_report,
)
from src.support_signal_service import (
    FileSupportSignalCache,
    SupportSignalService,
    SupportSignalServiceConfig,
)
from src.support_text import (
    SUPPORT_TEXT_CONTRACT_VERSION,
)


class DisabledExtractor:
    def __init__(self) -> None:
        self.calls = 0
        self.config = (
            SimpleNamespace(
                model="gpt-test"
            )
        )

    def generate(
        self,
        compiled: Any,
    ):
        self.calls += 1

        raise (
            SupportSignalGenerationDisabledError(
                "disabled for deterministic test"
            )
        )


def _write_input_tables(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
]:
    scoring = pd.DataFrame(
        [
            {
                "payment_id": "pay_001",
                "scoring_at": datetime(
                    2025,
                    6,
                    8,
                    tzinfo=timezone.utc,
                ),
            },
            {
                "payment_id": "pay_002",
                "scoring_at": datetime(
                    2025,
                    6,
                    8,
                    tzinfo=timezone.utc,
                ),
            },
            {
                "payment_id": "pay_003",
                "scoring_at": datetime(
                    2025,
                    6,
                    8,
                    tzinfo=timezone.utc,
                ),
            },
        ]
    )

    events = pd.DataFrame(
        [
            {
                "support_event_id": "sup_001",
                "payment_id": "pay_001",
                "event_at": datetime(
                    2025,
                    6,
                    5,
                    tzinfo=timezone.utc,
                ),
                "channel": "chat",
                "message_text": (
                    "Please cancel this order."
                ),
            },
            {
                "support_event_id": "sup_002",
                "payment_id": "pay_001",
                "event_at": datetime(
                    2025,
                    6,
                    20,
                    tzinfo=timezone.utc,
                ),
                "channel": "chat",
                "message_text": (
                    "The item has not arrived."
                ),
            },
            {
                "support_event_id": "sup_003",
                "payment_id": "pay_002",
                "event_at": datetime(
                    2025,
                    6,
                    4,
                    tzinfo=timezone.utc,
                ),
                "channel": "email",
                "message_text": (
                    "Please send the invoice."
                ),
            },
        ]
    )

    events_path = (
        tmp_path
        / "events.parquet"
    )

    scoring_path = (
        tmp_path
        / "scoring.parquet"
    )

    events.to_parquet(
        events_path,
        index=False,
    )

    scoring.to_parquet(
        scoring_path,
        index=False,
    )

    return events_path, scoring_path


def _service(
    tmp_path: Path,
) -> tuple[
    SupportSignalService,
    DisabledExtractor,
]:
    extractor = (
        DisabledExtractor()
    )

    cache = FileSupportSignalCache(
        tmp_path
        / "cache",
        now_factory=lambda: datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
    )

    service = SupportSignalService(
        extractor=extractor,
        cache=cache,
        config=(
            SupportSignalServiceConfig(
                generation_version=(
                    "report-test-v1"
                ),
                cache_ttl_seconds=3_600,
                fallback_cache_ttl_seconds=300,
            )
        ),
    )

    return service, extractor


def test_report_has_one_aligned_row_per_payment(
    tmp_path: Path,
) -> None:
    events_path, scoring_path = (
        _write_input_tables(
            tmp_path
        )
    )

    service, _ = _service(
        tmp_path
    )

    report = (
        generate_support_signal_report(
            events_path=events_path,
            scoring_path=scoring_path,
            output_path=(
                tmp_path
                / "report.parquet"
            ),
            service=service,
        )
    )

    assert report[
        "payment_id"
    ].tolist() == [
        "pay_001",
        "pay_002",
        "pay_003",
    ]

    assert report[
        "payment_id"
    ].is_unique

    assert len(report) == 3


def test_report_contains_exact_signal_columns(
    tmp_path: Path,
) -> None:
    events_path, scoring_path = (
        _write_input_tables(
            tmp_path
        )
    )

    service, _ = _service(
        tmp_path
    )

    report = (
        generate_support_signal_report(
            events_path=events_path,
            scoring_path=scoring_path,
            output_path=(
                tmp_path
                / "report.parquet"
            ),
            service=service,
        )
    )

    signal_columns = {
        "intent_to_cancel",
        "non_receipt_complaint",
        "dissatisfaction",
    }

    assert signal_columns.issubset(
        set(report.columns)
    )

    for column in signal_columns:
        assert (
            str(report[column].dtype)
            == "bool"
        )


def test_future_event_does_not_affect_signals(
    tmp_path: Path,
) -> None:
    events_path, scoring_path = (
        _write_input_tables(
            tmp_path
        )
    )

    service, _ = _service(
        tmp_path
    )

    report = (
        generate_support_signal_report(
            events_path=events_path,
            scoring_path=scoring_path,
            output_path=(
                tmp_path
                / "report.parquet"
            ),
            service=service,
        )
    )

    first = report.loc[
        report["payment_id"]
        == "pay_001"
    ].iloc[0]

    assert (
        bool(first["intent_to_cancel"])
        is True
    )

    assert (
        bool(
            first[
                "non_receipt_complaint"
            ]
        )
        is False
    )

    assert (
        first[
            "excluded_future_event_count"
        ]
        == 1
    )


def test_neutral_and_empty_rows_are_false(
    tmp_path: Path,
) -> None:
    events_path, scoring_path = (
        _write_input_tables(
            tmp_path
        )
    )

    service, _ = _service(
        tmp_path
    )

    report = (
        generate_support_signal_report(
            events_path=events_path,
            scoring_path=scoring_path,
            output_path=(
                tmp_path
                / "report.parquet"
            ),
            service=service,
        )
    )

    for payment_id in (
        "pay_002",
        "pay_003",
    ):
        row = report.loc[
            report["payment_id"]
            == payment_id
        ].iloc[0]

        assert not any(
            bool(row[column])
            for column in (
                "intent_to_cancel",
                "non_receipt_complaint",
                "dissatisfaction",
            )
        )


def test_report_contains_no_raw_text_or_outcomes(
    tmp_path: Path,
) -> None:
    events_path, scoring_path = (
        _write_input_tables(
            tmp_path
        )
    )

    service, _ = _service(
        tmp_path
    )

    report = (
        generate_support_signal_report(
            events_path=events_path,
            scoring_path=scoring_path,
            output_path=(
                tmp_path
                / "report.parquet"
            ),
            service=service,
        )
    )

    assert not (
        set(report.columns)
        & FORBIDDEN_REPORT_COLUMNS
    )

    serialized = report.to_json()

    assert "Please cancel" not in serialized
    assert "item has not arrived" not in serialized
    assert "customer_id" not in serialized
    assert (
        "chargeback_within_120d"
        not in serialized
    )
    assert "dispute_status" not in serialized
    assert "reason_code" not in serialized


def test_versions_and_provenance_are_recorded(
    tmp_path: Path,
) -> None:
    events_path, scoring_path = (
        _write_input_tables(
            tmp_path
        )
    )

    service, _ = _service(
        tmp_path
    )

    report = (
        generate_support_signal_report(
            events_path=events_path,
            scoring_path=scoring_path,
            output_path=(
                tmp_path
                / "report.parquet"
            ),
            service=service,
        )
    )

    assert set(
        report[
            "support_text_contract_version"
        ]
    ) == {
        SUPPORT_TEXT_CONTRACT_VERSION
    }

    assert set(
        report["report_version"]
    ) == {
        SUPPORT_SIGNAL_REPORT_VERSION
    }

    assert report[
        "fallback_used"
    ].all()

    assert set(
        report["provider"]
    ) == {
        "deterministic_fallback"
    }


def test_report_file_matches_returned_frame(
    tmp_path: Path,
) -> None:
    events_path, scoring_path = (
        _write_input_tables(
            tmp_path
        )
    )

    output_path = (
        tmp_path
        / "report.parquet"
    )

    service, _ = _service(
        tmp_path
    )

    returned = (
        generate_support_signal_report(
            events_path=events_path,
            scoring_path=scoring_path,
            output_path=output_path,
            service=service,
        )
    )

    saved = pd.read_parquet(
        output_path
    )

    pd.testing.assert_frame_equal(
        returned,
        saved,
        check_exact=True,
    )


def test_in_memory_hash_cache_reduces_calls(
    tmp_path: Path,
) -> None:
    events_path, scoring_path = (
        _write_input_tables(
            tmp_path
        )
    )

    service, extractor = _service(
        tmp_path
    )

    report = (
        generate_support_signal_report(
            events_path=events_path,
            scoring_path=scoring_path,
            output_path=(
                tmp_path
                / "report.parquet"
            ),
            service=service,
        )
    )

    assert (
        len(report)
        == 3
    )

    assert (
        extractor.calls
        <= report[
            "input_hash"
        ].nunique()
    )