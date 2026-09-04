from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from src.support_events import (
    FORBIDDEN_OUTPUT_COLUMNS,
    GENERATION_VERSION,
    SCORING_POLICY_VERSION,
    SCORING_WINDOW_DAYS,
    SOURCE_COLUMNS,
    generate_support_dataset,
)


def _source_frame(
    *,
    repetitions: int = 200,
) -> pd.DataFrame:
    archetypes = [
        "NORMAL",
        "CONFUSED_LEGIT",
        "SERIAL_DISPUTER",
        "DISSATISFIED",
        "STOLEN_CARD_USER",
    ]

    rows: list[
        dict[str, object]
    ] = []

    total = (
        repetitions
        * len(archetypes)
    )

    timestamps = pd.date_range(
        "2025-01-01",
        periods=total,
        freq="h",
        tz="UTC",
    )

    index = 0

    for repetition in range(
        repetitions
    ):
        for archetype in archetypes:
            rows.append(
                {
                    "payment_id": (
                        f"pay_test_{index:06d}"
                    ),
                    "archetype": archetype,
                    "created_at": (
                        timestamps[index]
                    ),
                    # These columns deliberately exist in
                    # the source but must never be loaded
                    # into or written by the generator.
                    "customer_id": (
                        f"cust_secret_{index}"
                    ),
                    "chargeback_within_120d": (
                        index % 2
                    ),
                    "dispute_status": "lost",
                    "reason_code": "10.4",
                    "true_fraud": True,
                }
            )

            index += 1

    return pd.DataFrame(
        rows
    )


def _generate(
    tmp_path: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    source = _source_frame()

    source_path = (
        tmp_path
        / "raw_payments.parquet"
    )

    events_path = (
        tmp_path
        / "support_events.parquet"
    )

    scoring_path = (
        tmp_path
        / "support_scoring.parquet"
    )

    source.to_parquet(
        source_path,
        index=False,
    )

    events, scoring = (
        generate_support_dataset(
            source_path=source_path,
            events_path=events_path,
            scoring_path=scoring_path,
        )
    )

    return source, events, scoring


def test_source_column_allowlist_excludes_outcomes() -> None:
    assert SOURCE_COLUMNS == [
        "payment_id",
        "archetype",
        "created_at",
    ]

    assert (
        set(SOURCE_COLUMNS)
        & FORBIDDEN_OUTPUT_COLUMNS
        == {"archetype"}
    )

    assert "customer_id" not in SOURCE_COLUMNS
    assert (
        "chargeback_within_120d"
        not in SOURCE_COLUMNS
    )
    assert "dispute_status" not in SOURCE_COLUMNS
    assert "reason_code" not in SOURCE_COLUMNS
    assert "true_fraud" not in SOURCE_COLUMNS


def test_output_files_are_created(
    tmp_path: Path,
) -> None:
    _generate(
        tmp_path
    )

    assert (
        tmp_path
        / "support_events.parquet"
    ).exists()

    assert (
        tmp_path
        / "support_scoring.parquet"
    ).exists()


def test_scoring_table_covers_every_payment(
    tmp_path: Path,
) -> None:
    source, _, scoring = _generate(
        tmp_path
    )

    assert len(scoring) == len(source)
    assert scoring["payment_id"].is_unique

    assert (
        scoring["payment_id"].tolist()
        == source["payment_id"].tolist()
    )


def test_scoring_timestamp_is_exactly_seven_days(
    tmp_path: Path,
) -> None:
    source, _, scoring = _generate(
        tmp_path
    )

    expected = (
        pd.to_datetime(
            source["created_at"],
            utc=True,
        )
        + pd.Timedelta(
            days=SCORING_WINDOW_DAYS
        )
    )

    actual = pd.to_datetime(
        scoring["scoring_at"],
        utc=True,
    )

    pd.testing.assert_series_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )

    assert set(
        scoring[
            "scoring_policy_version"
        ]
    ) == {
        SCORING_POLICY_VERSION
    }


def test_events_have_expected_schema(
    tmp_path: Path,
) -> None:
    _, events, _ = _generate(
        tmp_path
    )

    assert list(events.columns) == [
        "support_event_id",
        "payment_id",
        "event_at",
        "channel",
        "message_text",
        "generation_version",
    ]

    assert events[
        "support_event_id"
    ].is_unique

    assert not events[
        "payment_id"
    ].isna().any()

    assert not events[
        "message_text"
    ].isna().any()

    assert set(
        events["generation_version"]
    ) == {
        GENERATION_VERSION
    }


def test_outputs_contain_no_forbidden_fields(
    tmp_path: Path,
) -> None:
    _, events, scoring = _generate(
        tmp_path
    )

    assert not (
        set(events.columns)
        & FORBIDDEN_OUTPUT_COLUMNS
    )

    assert not (
        set(scoring.columns)
        & FORBIDDEN_OUTPUT_COLUMNS
    )


def test_events_reference_only_known_payments(
    tmp_path: Path,
) -> None:
    source, events, _ = _generate(
        tmp_path
    )

    assert set(
        events["payment_id"]
    ).issubset(
        set(source["payment_id"])
    )


def test_events_include_observable_and_future_rows(
    tmp_path: Path,
) -> None:
    _, events, scoring = _generate(
        tmp_path
    )

    merged = events.merge(
        scoring[
            [
                "payment_id",
                "scoring_at",
            ]
        ],
        on="payment_id",
        how="left",
        validate="many_to_one",
    )

    assert (
        merged["event_at"]
        <= merged["scoring_at"]
    ).any()

    assert (
        merged["event_at"]
        > merged["scoring_at"]
    ).any()


def test_all_events_occur_after_payment(
    tmp_path: Path,
) -> None:
    source, events, _ = _generate(
        tmp_path
    )

    merged = events.merge(
        source[
            [
                "payment_id",
                "created_at",
            ]
        ],
        on="payment_id",
        how="left",
        validate="many_to_one",
    )

    assert (
        merged["event_at"]
        > merged["created_at"]
    ).all()


def test_support_text_contains_no_pii_or_injection(
    tmp_path: Path,
) -> None:
    _, events, _ = _generate(
        tmp_path
    )

    all_text = " ".join(
        events[
            "message_text"
        ].astype(str)
    )

    assert not re.search(
        r"\b[A-Z0-9._%+-]+"
        r"@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        all_text,
        flags=re.IGNORECASE,
    )

    assert not re.search(
        r"(?<!\d)(?:\+91[-\s]?)?"
        r"[6-9]\d{9}(?!\d)",
        all_text,
    )

    assert not re.search(
        r"\bignore\s+(?:all\s+)?"
        r"(?:previous|prior|above)\b",
        all_text,
        flags=re.IGNORECASE,
    )

    assert "cust_secret_" not in all_text
    assert "10.4" not in all_text


def test_generation_is_deterministic(
    tmp_path: Path,
) -> None:
    source = _source_frame()

    source_path = (
        tmp_path
        / "raw_payments.parquet"
    )

    source.to_parquet(
        source_path,
        index=False,
    )

    first_events, first_scoring = (
        generate_support_dataset(
            source_path=source_path,
            events_path=(
                tmp_path
                / "first_events.parquet"
            ),
            scoring_path=(
                tmp_path
                / "first_scoring.parquet"
            ),
        )
    )

    second_events, second_scoring = (
        generate_support_dataset(
            source_path=source_path,
            events_path=(
                tmp_path
                / "second_events.parquet"
            ),
            scoring_path=(
                tmp_path
                / "second_scoring.parquet"
            ),
        )
    )

    pd.testing.assert_frame_equal(
        first_events,
        second_events,
        check_exact=True,
    )

    pd.testing.assert_frame_equal(
        first_scoring,
        second_scoring,
        check_exact=True,
    )


def test_duplicate_payment_ids_are_rejected(
    tmp_path: Path,
) -> None:
    source = _source_frame(
        repetitions=2
    )

    source.loc[
        1,
        "payment_id",
    ] = source.loc[
        0,
        "payment_id",
    ]

    source_path = (
        tmp_path
        / "duplicates.parquet"
    )

    source.to_parquet(
        source_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="duplicate payment IDs",
    ):
        generate_support_dataset(
            source_path=source_path,
            events_path=(
                tmp_path
                / "events.parquet"
            ),
            scoring_path=(
                tmp_path
                / "scoring.parquet"
            ),
        )


def test_unknown_archetype_is_rejected(
    tmp_path: Path,
) -> None:
    source = _source_frame(
        repetitions=2
    )

    source.loc[
        0,
        "archetype",
    ] = "UNKNOWN_ARCHETYPE"

    source_path = (
        tmp_path
        / "unknown.parquet"
    )

    source.to_parquet(
        source_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported archetypes",
    ):
        generate_support_dataset(
            source_path=source_path,
            events_path=(
                tmp_path
                / "events.parquet"
            ),
            scoring_path=(
                tmp_path
                / "scoring.parquet"
            ),
        )