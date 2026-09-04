from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Final

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SOURCE_PATH = (
    ROOT
    / "data"
    / "raw_payments.parquet"
)

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

GENERATION_VERSION: Final = (
    "support-events-v1"
)

SCORING_POLICY_VERSION: Final = (
    "payment-plus-7d-v1"
)

SCORING_WINDOW_DAYS: Final = 7


SOURCE_COLUMNS: Final = [
    "payment_id",
    "archetype",
    "created_at",
]


FORBIDDEN_OUTPUT_COLUMNS: Final = {
    "customer_id",
    "customer_name",
    "customer_email",
    "customer_phone",
    "archetype",
    "chargeback_within_120d",
    "chargeback_outcome",
    "true_fraud",
    "dispute_id",
    "dispute_status",
    "reason_code",
    "chargeback_family",
    "respond_by",
}


SUPPORTED_ARCHETYPES: Final = {
    "NORMAL",
    "CONFUSED_LEGIT",
    "SERIAL_DISPUTER",
    "DISSATISFIED",
    "STOLEN_CARD_USER",
}


# These probabilities simulate whether a support interaction occurs.
# They do not use or inspect the final chargeback outcome.
PRE_SCORING_EVENT_PROBABILITY: Final = {
    "NORMAL": 0.025,
    "CONFUSED_LEGIT": 0.45,
    "SERIAL_DISPUTER": 0.28,
    "DISSATISFIED": 0.68,
    "STOLEN_CARD_USER": 0.04,
}


# Future events are deliberately retained in the raw support-event table.
# The extractor must prove that these events never cross the scoring boundary.
POST_SCORING_EVENT_PROBABILITY: Final = {
    "NORMAL": 0.04,
    "CONFUSED_LEGIT": 0.16,
    "SERIAL_DISPUTER": 0.18,
    "DISSATISFIED": 0.22,
    "STOLEN_CARD_USER": 0.08,
}


CATEGORY_WEIGHTS: Final = {
    "NORMAL": {
        "intent_to_cancel": 0.05,
        "non_receipt_complaint": 0.08,
        "dissatisfaction": 0.07,
        "neutral": 0.80,
    },
    "CONFUSED_LEGIT": {
        "intent_to_cancel": 0.10,
        "non_receipt_complaint": 0.50,
        "dissatisfaction": 0.18,
        "neutral": 0.22,
    },
    "SERIAL_DISPUTER": {
        "intent_to_cancel": 0.18,
        "non_receipt_complaint": 0.22,
        "dissatisfaction": 0.40,
        "neutral": 0.20,
    },
    "DISSATISFIED": {
        "intent_to_cancel": 0.20,
        "non_receipt_complaint": 0.25,
        "dissatisfaction": 0.50,
        "neutral": 0.05,
    },
    "STOLEN_CARD_USER": {
        "intent_to_cancel": 0.02,
        "non_receipt_complaint": 0.03,
        "dissatisfaction": 0.05,
        "neutral": 0.90,
    },
}


MESSAGE_TEMPLATES: Final = {
    "intent_to_cancel": (
        (
            "Please cancel this order. "
            "I do not want it to continue."
        ),
        (
            "I want to cancel this purchase "
            "before it proceeds further."
        ),
        (
            "Please stop the subscription from "
            "continuing into another billing cycle."
        ),
    ),
    "non_receipt_complaint": (
        (
            "The order has not arrived yet. "
            "Please check the delivery status."
        ),
        (
            "I still have not received the item "
            "connected with this payment."
        ),
        (
            "The expected delivery has not reached "
            "me. Please provide an update."
        ),
    ),
    "dissatisfaction": (
        (
            "I am unhappy with the product and "
            "the service received."
        ),
        (
            "The purchase did not meet my "
            "expectations and I am dissatisfied."
        ),
        (
            "I am not satisfied with the quality "
            "of the order."
        ),
    ),
    "neutral": (
        (
            "Please send me a copy of the invoice "
            "for this payment."
        ),
        (
            "Can you confirm the order reference "
            "for this purchase?"
        ),
        (
            "I would like general information "
            "about this transaction."
        ),
    ),
}


CHANNELS: Final = (
    "chat",
    "email",
    "phone_transcript",
)


def _stable_unit(
    *parts: object,
) -> float:
    material = "|".join(
        str(part)
        for part in parts
    ).encode("utf-8")

    digest = sha256(
        material
    ).digest()

    integer = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    return integer / float(2**64)


def _stable_integer(
    minimum: int,
    maximum: int,
    *parts: object,
) -> int:
    if minimum > maximum:
        raise ValueError(
            "minimum cannot exceed maximum"
        )

    width = (
        maximum
        - minimum
        + 1
    )

    return minimum + int(
        _stable_unit(*parts)
        * width
    )


def _choose_category(
    archetype: str,
    *,
    payment_id: str,
    timing: str,
) -> str:
    weights = CATEGORY_WEIGHTS[
        archetype
    ]

    draw = _stable_unit(
        GENERATION_VERSION,
        payment_id,
        timing,
        "category",
    )

    cumulative = 0.0

    for category, weight in (
        weights.items()
    ):
        cumulative += weight

        if draw < cumulative:
            return category

    return "neutral"


def _choose_message(
    category: str,
    *,
    payment_id: str,
    timing: str,
) -> str:
    templates = MESSAGE_TEMPLATES[
        category
    ]

    index = _stable_integer(
        0,
        len(templates) - 1,
        GENERATION_VERSION,
        payment_id,
        timing,
        "template",
    )

    return templates[index]


def _choose_channel(
    *,
    payment_id: str,
    timing: str,
) -> str:
    index = _stable_integer(
        0,
        len(CHANNELS) - 1,
        GENERATION_VERSION,
        payment_id,
        timing,
        "channel",
    )

    return CHANNELS[index]


def _event_id(
    *,
    payment_id: str,
    timing: str,
) -> str:
    digest = sha256(
        (
            GENERATION_VERSION
            + "|"
            + payment_id
            + "|"
            + timing
        ).encode("utf-8")
    ).hexdigest()[:20]

    return f"sup_{digest}"


def _load_source(
    source_path: Path = SOURCE_PATH,
) -> pd.DataFrame:
    if not source_path.exists():
        raise FileNotFoundError(
            "Raw payment source is missing: "
            f"{source_path}"
        )

    # Only these three columns are read. Outcome, dispute and
    # customer-identity fields cannot enter this generator.
    source = pd.read_parquet(
        source_path,
        columns=SOURCE_COLUMNS,
    )

    missing = sorted(
        set(SOURCE_COLUMNS)
        - set(source.columns)
    )

    if missing:
        raise ValueError(
            "Support-event source is missing "
            "required columns: "
            + ", ".join(missing)
        )

    if source.empty:
        raise ValueError(
            "Support-event source contains "
            "zero rows"
        )

    if source[
        "payment_id"
    ].isna().any():
        raise ValueError(
            "Source contains missing payment_id"
        )

    source = source.copy()

    source["payment_id"] = (
        source["payment_id"]
        .astype(str)
    )

    source["archetype"] = (
        source["archetype"]
        .astype(str)
    )

    if source[
        "payment_id"
    ].duplicated().any():
        raise ValueError(
            "Source contains duplicate "
            "payment IDs"
        )

    unknown_archetypes = sorted(
        set(source["archetype"])
        - SUPPORTED_ARCHETYPES
    )

    if unknown_archetypes:
        raise ValueError(
            "Unsupported archetypes: "
            + ", ".join(
                unknown_archetypes
            )
        )

    created_at = pd.to_datetime(
        source["created_at"],
        errors="coerce",
    )

    if created_at.isna().any():
        raise ValueError(
            "Source contains invalid "
            "created_at timestamps"
        )

    if created_at.dt.tz is None:
        raise ValueError(
            "created_at timestamps must be "
            "timezone-aware"
        )

    source["created_at"] = (
        created_at.dt.tz_convert(
            "UTC"
        )
    )

    return source


def _build_event(
    *,
    payment_id: str,
    archetype: str,
    created_at: pd.Timestamp,
    timing: str,
) -> dict[str, object]:
    if timing == "before_scoring":
        day_offset = _stable_integer(
            1,
            SCORING_WINDOW_DAYS - 1,
            GENERATION_VERSION,
            payment_id,
            timing,
            "day",
        )
    elif timing == "after_scoring":
        day_offset = _stable_integer(
            SCORING_WINDOW_DAYS + 1,
            30,
            GENERATION_VERSION,
            payment_id,
            timing,
            "day",
        )
    else:
        raise ValueError(
            f"Unsupported event timing: {timing}"
        )

    second_offset = _stable_integer(
        0,
        86_399,
        GENERATION_VERSION,
        payment_id,
        timing,
        "second",
    )

    event_at = (
        created_at
        + pd.Timedelta(
            days=day_offset,
            seconds=second_offset,
        )
    )

    category = _choose_category(
        archetype,
        payment_id=payment_id,
        timing=timing,
    )

    return {
        "support_event_id": _event_id(
            payment_id=payment_id,
            timing=timing,
        ),
        "payment_id": payment_id,
        "event_at": event_at,
        "channel": _choose_channel(
            payment_id=payment_id,
            timing=timing,
        ),
        "message_text": _choose_message(
            category,
            payment_id=payment_id,
            timing=timing,
        ),
        "generation_version": (
            GENERATION_VERSION
        ),
    }


def generate_support_dataset(
    *,
    source_path: Path = SOURCE_PATH,
    events_path: Path = SUPPORT_EVENTS_PATH,
    scoring_path: Path = SUPPORT_SCORING_PATH,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    source = _load_source(
        source_path
    )

    scoring = pd.DataFrame(
        {
            "payment_id": (
                source["payment_id"]
                .astype(str)
            ),
            "scoring_at": (
                source["created_at"]
                + pd.Timedelta(
                    days=SCORING_WINDOW_DAYS
                )
            ),
            "scoring_policy_version": (
                SCORING_POLICY_VERSION
            ),
        }
    )

    event_records: list[
        dict[str, object]
    ] = []

    for row in source.itertuples(
        index=False
    ):
        payment_id = str(
            row.payment_id
        )

        archetype = str(
            row.archetype
        )

        created_at = pd.Timestamp(
            row.created_at
        )

        pre_draw = _stable_unit(
            GENERATION_VERSION,
            payment_id,
            "before_scoring",
            "occurs",
        )

        if (
            pre_draw
            < PRE_SCORING_EVENT_PROBABILITY[
                archetype
            ]
        ):
            event_records.append(
                _build_event(
                    payment_id=payment_id,
                    archetype=archetype,
                    created_at=created_at,
                    timing="before_scoring",
                )
            )

        post_draw = _stable_unit(
            GENERATION_VERSION,
            payment_id,
            "after_scoring",
            "occurs",
        )

        if (
            post_draw
            < POST_SCORING_EVENT_PROBABILITY[
                archetype
            ]
        ):
            event_records.append(
                _build_event(
                    payment_id=payment_id,
                    archetype=archetype,
                    created_at=created_at,
                    timing="after_scoring",
                )
            )

    event_columns = [
        "support_event_id",
        "payment_id",
        "event_at",
        "channel",
        "message_text",
        "generation_version",
    ]

    events = pd.DataFrame.from_records(
        event_records,
        columns=event_columns,
    )

    if not events.empty:
        events["event_at"] = (
            pd.to_datetime(
                events["event_at"],
                utc=True,
            )
        )

        events = (
            events.sort_values(
                [
                    "payment_id",
                    "event_at",
                    "support_event_id",
                ],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

    if events[
        "support_event_id"
    ].duplicated().any():
        raise AssertionError(
            "Generated support event IDs "
            "are not unique"
        )

    if not set(
        events["payment_id"]
    ).issubset(
        set(scoring["payment_id"])
    ):
        raise AssertionError(
            "Support events contain unknown "
            "payment IDs"
        )

    for frame in (
        events,
        scoring,
    ):
        if (
            set(frame.columns)
            & FORBIDDEN_OUTPUT_COLUMNS
        ):
            raise AssertionError(
                "Forbidden fields reached a "
                "support output"
            )

    events_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scoring_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    events.to_parquet(
        events_path,
        index=False,
    )

    scoring.to_parquet(
        scoring_path,
        index=False,
    )

    return events, scoring


def main() -> None:
    events, scoring = (
        generate_support_dataset()
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

    before_count = int(
        (
            merged["event_at"]
            <= merged["scoring_at"]
        ).sum()
    )

    future_count = int(
        (
            merged["event_at"]
            > merged["scoring_at"]
        ).sum()
    )

    print(
        "Synthetic support dataset generated"
    )
    print(
        f"Payments with scoring timestamps: "
        f"{len(scoring):,}"
    )
    print(
        f"Support events: {len(events):,}"
    )
    print(
        f"Observable at scoring: "
        f"{before_count:,}"
    )
    print(
        f"After scoring and excluded later: "
        f"{future_count:,}"
    )
    print(
        "Events output:",
        SUPPORT_EVENTS_PATH.relative_to(
            ROOT
        ),
    )
    print(
        "Scoring output:",
        SUPPORT_SCORING_PATH.relative_to(
            ROOT
        ),
    )


if __name__ == "__main__":
    main()