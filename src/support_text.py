from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Final, Literal

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
)


ROOT = Path(__file__).resolve().parents[1]

SUPPORT_EVENTS_PATH = (
    ROOT
    / "data"
    / "support_events.parquet"
)

SUPPORT_SCORING_PATH = (
    ROOT
    / "data"
    / "support_scoring.parquet"
)

SUPPORT_TEXT_CONTRACT_VERSION: Final = (
    "support-text-v1"
)

MAX_EVENTS_PER_MODEL_REQUEST: Final = 10


SIGNAL_NAMES: Final = (
    "intent_to_cancel",
    "non_receipt_complaint",
    "dissatisfaction",
)


EVENT_COLUMNS: Final = [
    "support_event_id",
    "payment_id",
    "event_at",
    "channel",
    "message_text",
]


SCORING_COLUMNS: Final = [
    "payment_id",
    "scoring_at",
]


PROMPT_INJECTION_PATTERNS: Final = (
    r"\bignore\s+(?:all\s+)?"
    r"(?:previous|prior|above)\b",
    r"\b(?:system|developer)\s+"
    r"(?:prompt|message|instructions?)\b",
    r"\b(?:reveal|print|show)\s+"
    r"(?:the\s+)?(?:prompt|instructions?)\b",
    r"\bjailbreak\b",
    r"\bdo\s+not\s+follow\s+"
    r"(?:the\s+)?instructions?\b",
    r"<\s*/?\s*"
    r"(?:system|assistant|developer)"
    r"\s*>",
)


EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+"
    r"@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    flags=re.IGNORECASE,
)

CARD_PATTERN = re.compile(
    r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"
)

INDIAN_PHONE_PATTERN = re.compile(
    r"(?<!\d)"
    r"(?:\+91[-\s]?)?"
    r"[6-9]\d{9}"
    r"(?!\d)"
)

CUSTOMER_ID_PATTERN = re.compile(
    r"\bcust_[A-Za-z0-9_-]+\b",
    flags=re.IGNORECASE,
)


class SupportTextSafetyError(ValueError):
    pass


class SafeSupportEvent(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    event_ref: str = Field(
        pattern=r"^event_\d{3}$",
    )

    observed_at: datetime

    channel: Literal[
        "chat",
        "email",
        "phone_transcript",
    ]

    text: str = Field(
        min_length=1,
        max_length=2_000,
    )

    @field_validator(
        "observed_at"
    )
    @classmethod
    def validate_observed_at(
        cls,
        value: datetime,
    ) -> datetime:
        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "observed_at must be "
                "timezone-aware"
            )

        return value


class SupportSignals(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    intent_to_cancel: StrictBool

    non_receipt_complaint: StrictBool

    dissatisfaction: StrictBool


class CompiledSupportText(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    # Internal lookup field. It is deliberately excluded
    # by model_payload().
    payment_id: str = Field(
        min_length=1,
    )

    # Internal enforcement timestamp. It is deliberately
    # excluded by model_payload().
    scoring_at: datetime

    events: list[
        SafeSupportEvent
    ] = Field(
        max_length=(
            MAX_EVENTS_PER_MODEL_REQUEST
        )
    )

    input_hash: str = Field(
        pattern=r"^[a-f0-9]{64}$",
    )

    source_event_count: int = Field(
        ge=0,
    )

    observable_event_count: int = Field(
        ge=0,
    )

    excluded_future_event_count: int = Field(
        ge=0,
    )

    excluded_injection_event_count: int = Field(
        ge=0,
    )

    excluded_overflow_event_count: int = Field(
        ge=0,
    )

    pii_redaction_count: int = Field(
        ge=0,
    )

    @field_validator(
        "scoring_at"
    )
    @classmethod
    def validate_scoring_at(
        cls,
        value: datetime,
    ) -> datetime:
        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "scoring_at must be "
                "timezone-aware"
            )

        return value

    def model_payload(
        self,
    ) -> dict[str, Any]:
        """
        Return the only payload permitted to cross the model boundary.

        Payment IDs, source event IDs, absolute timestamps, customer
        identity, outcomes and dispute fields are intentionally absent.
        """
        return {
            "contract_version": (
                SUPPORT_TEXT_CONTRACT_VERSION
            ),
            "events": [
                {
                    "event_ref": (
                        event.event_ref
                    ),
                    "channel": (
                        event.channel
                    ),
                    "text": event.text,
                }
                for event in self.events
            ],
        }


def contains_prompt_injection(
    text: str,
) -> bool:
    return any(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        for pattern in (
            PROMPT_INJECTION_PATTERNS
        )
    )


def sanitize_support_text(
    text: str,
) -> tuple[str, int]:
    normalized = " ".join(
        str(text).split()
    ).strip()

    if not normalized:
        raise SupportTextSafetyError(
            "Support text is empty"
        )

    redaction_count = 0

    normalized, count = (
        CARD_PATTERN.subn(
            "[REDACTED_CARD]",
            normalized,
        )
    )
    redaction_count += count

    normalized, count = (
        EMAIL_PATTERN.subn(
            "[REDACTED_EMAIL]",
            normalized,
        )
    )
    redaction_count += count

    normalized, count = (
        INDIAN_PHONE_PATTERN.subn(
            "[REDACTED_PHONE]",
            normalized,
        )
    )
    redaction_count += count

    normalized, count = (
        CUSTOMER_ID_PATTERN.subn(
            "[REDACTED_CUSTOMER_ID]",
            normalized,
        )
    )
    redaction_count += count

    return normalized, redaction_count


def _normalize_utc_timestamp(
    value: object,
    *,
    field_name: str,
) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(
            value
        )
    except Exception as error:
        raise SupportTextSafetyError(
            f"{field_name} is invalid"
        ) from error

    if timestamp is pd.NaT:
        raise SupportTextSafetyError(
            f"{field_name} is missing"
        )

    if (
        timestamp.tzinfo is None
        or timestamp.utcoffset() is None
    ):
        raise SupportTextSafetyError(
            f"{field_name} must be "
            "timezone-aware"
        )

    return timestamp.tz_convert(
        "UTC"
    )


def _model_payload_for_events(
    events: list[
        SafeSupportEvent
    ],
) -> dict[str, Any]:
    return {
        "contract_version": (
            SUPPORT_TEXT_CONTRACT_VERSION
        ),
        "events": [
            {
                "event_ref": (
                    event.event_ref
                ),
                "channel": (
                    event.channel
                ),
                "text": event.text,
            }
            for event in events
        ],
    }


def _hash_payload(
    payload: dict[str, Any],
) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def compile_support_text(
    *,
    payment_id: str,
    scoring_at: object,
    events: pd.DataFrame,
) -> CompiledSupportText:
    if not str(payment_id).strip():
        raise ValueError(
            "payment_id cannot be empty"
        )

    missing_columns = sorted(
        set(EVENT_COLUMNS)
        - set(events.columns)
    )

    if missing_columns:
        raise ValueError(
            "Support events are missing "
            "required columns: "
            + ", ".join(
                missing_columns
            )
        )

    normalized_scoring_at = (
        _normalize_utc_timestamp(
            scoring_at,
            field_name="scoring_at",
        )
    )

    selected = events.loc[
        events["payment_id"].astype(str)
        == str(payment_id)
    ].copy()

    if selected[
        "support_event_id"
    ].duplicated().any():
        raise ValueError(
            "Support events contain duplicate "
            "event IDs for this payment"
        )

    source_event_count = len(
        selected
    )

    normalized_rows: list[
        dict[str, object]
    ] = []

    for row in selected.itertuples(
        index=False
    ):
        event_at = (
            _normalize_utc_timestamp(
                row.event_at,
                field_name="event_at",
            )
        )

        normalized_rows.append(
            {
                "support_event_id": str(
                    row.support_event_id
                ),
                "event_at": event_at,
                "channel": str(
                    row.channel
                ),
                "message_text": str(
                    row.message_text
                ),
            }
        )

    normalized_rows.sort(
        key=lambda item: (
            item["event_at"],
            item["support_event_id"],
        )
    )

    future_count = 0
    injection_count = 0
    pii_redaction_count = 0

    safe_rows: list[
        dict[str, object]
    ] = []

    for row in normalized_rows:
        event_at = row["event_at"]

        if event_at > normalized_scoring_at:
            future_count += 1
            continue

        raw_text = str(
            row["message_text"]
        )

        if contains_prompt_injection(
            raw_text
        ):
            injection_count += 1
            continue

        sanitized_text, redactions = (
            sanitize_support_text(
                raw_text
            )
        )

        pii_redaction_count += redactions

        safe_rows.append(
            {
                "observed_at": event_at,
                "channel": row["channel"],
                "text": sanitized_text,
            }
        )

    overflow_count = max(
        0,
        len(safe_rows)
        - MAX_EVENTS_PER_MODEL_REQUEST,
    )

    if overflow_count:
        safe_rows = safe_rows[
            -MAX_EVENTS_PER_MODEL_REQUEST:
        ]

    safe_events = [
        SafeSupportEvent(
            event_ref=(
                f"event_{index:03d}"
            ),
            observed_at=row[
                "observed_at"
            ],
            channel=row["channel"],
            text=row["text"],
        )
        for index, row in enumerate(
            safe_rows,
            start=1,
        )
    ]

    model_payload = (
        _model_payload_for_events(
            safe_events
        )
    )

    input_hash = _hash_payload(
        model_payload
    )

    return CompiledSupportText(
        payment_id=str(payment_id),
        scoring_at=(
            normalized_scoring_at.to_pydatetime()
        ),
        events=safe_events,
        input_hash=input_hash,
        source_event_count=(
            source_event_count
        ),
        observable_event_count=len(
            safe_events
        ),
        excluded_future_event_count=(
            future_count
        ),
        excluded_injection_event_count=(
            injection_count
        ),
        excluded_overflow_event_count=(
            overflow_count
        ),
        pii_redaction_count=(
            pii_redaction_count
        ),
    )


def _load_support_tables(
    *,
    events_path: Path = SUPPORT_EVENTS_PATH,
    scoring_path: Path = SUPPORT_SCORING_PATH,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    if not events_path.exists():
        raise FileNotFoundError(
            "Support events are missing: "
            f"{events_path}"
        )

    if not scoring_path.exists():
        raise FileNotFoundError(
            "Support scoring table is missing: "
            f"{scoring_path}"
        )

    events = pd.read_parquet(
        events_path,
        columns=EVENT_COLUMNS,
    )

    scoring = pd.read_parquet(
        scoring_path,
        columns=SCORING_COLUMNS,
    )

    if scoring.empty:
        raise ValueError(
            "Support scoring table contains "
            "zero rows"
        )

    if scoring[
        "payment_id"
    ].isna().any():
        raise ValueError(
            "Support scoring table contains "
            "missing payment IDs"
        )

    scoring = scoring.copy()

    scoring["payment_id"] = (
        scoring["payment_id"]
        .astype(str)
    )

    if scoring[
        "payment_id"
    ].duplicated().any():
        raise ValueError(
            "Support scoring table contains "
            "duplicate payment IDs"
        )

    if events[
        "support_event_id"
    ].duplicated().any():
        raise ValueError(
            "Support event table contains "
            "duplicate event IDs"
        )

    return events, scoring


def compile_payment_support_text(
    payment_id: str,
    *,
    events_path: Path = SUPPORT_EVENTS_PATH,
    scoring_path: Path = SUPPORT_SCORING_PATH,
) -> CompiledSupportText:
    events, scoring = (
        _load_support_tables(
            events_path=events_path,
            scoring_path=scoring_path,
        )
    )

    matches = scoring.loc[
        scoring["payment_id"]
        == str(payment_id)
    ]

    if matches.empty:
        raise KeyError(
            "Unknown support-scoring "
            f"payment_id: {payment_id}"
        )

    if len(matches) != 1:
        raise ValueError(
            "payment_id matched more than one "
            "support-scoring row"
        )

    return compile_support_text(
        payment_id=str(payment_id),
        scoring_at=(
            matches.iloc[0][
                "scoring_at"
            ]
        ),
        events=events,
    )