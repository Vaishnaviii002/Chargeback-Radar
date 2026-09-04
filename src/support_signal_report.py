from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from src.support_signal_service import (
    SupportSignalDeliveryResult,
    SupportSignalService,
)
from src.support_text import (
    EVENT_COLUMNS,
    SUPPORT_EVENTS_PATH,
    SUPPORT_SCORING_PATH,
    SUPPORT_TEXT_CONTRACT_VERSION,
    _load_support_tables,
    compile_support_text,
)


ROOT = Path(__file__).resolve().parents[1]

SUPPORT_SIGNAL_REPORT_PATH = (
    ROOT
    / "reports"
    / "support_signals.parquet"
)

SUPPORT_SIGNAL_REPORT_VERSION: Final = (
    "support-signal-report-v1"
)


FORBIDDEN_REPORT_COLUMNS: Final = {
    "customer_id",
    "customer_name",
    "customer_email",
    "customer_phone",
    "archetype",
    "message_text",
    "support_event_id",
    "chargeback_within_120d",
    "chargeback_outcome",
    "true_fraud",
    "dispute_id",
    "dispute_status",
    "reason_code",
    "chargeback_family",
    "respond_by",
}


def _reuse_delivery(
    cached: SupportSignalDeliveryResult,
    *,
    payment_id: str,
) -> SupportSignalDeliveryResult:
    fallback_cached = (
        cached.provider
        == "deterministic_fallback"
    )

    return cached.model_copy(
        update={
            "payment_id": payment_id,
            "delivery_mode": "CACHE",
            "latency_ms": 0,
            "cache_hit": True,
            "fallback_used": (
                fallback_cached
            ),
            "fallback_reason": (
                "CACHED_FALLBACK"
                if fallback_cached
                else None
            ),
        }
    )


def generate_support_signal_report(
    *,
    events_path: Path = SUPPORT_EVENTS_PATH,
    scoring_path: Path = SUPPORT_SCORING_PATH,
    output_path: Path = (
        SUPPORT_SIGNAL_REPORT_PATH
    ),
    service: (
        SupportSignalService
        | None
    ) = None,
) -> pd.DataFrame:
    events, scoring = (
        _load_support_tables(
            events_path=events_path,
            scoring_path=scoring_path,
        )
    )

    active_service = (
        service
        or SupportSignalService()
    )

    events_by_payment = {
        str(payment_id): group.copy()
        for payment_id, group
        in events.groupby(
            "payment_id",
            sort=False,
        )
    }

    empty_events = pd.DataFrame(
        columns=EVENT_COLUMNS
    )

    delivery_by_input_hash: dict[
        str,
        SupportSignalDeliveryResult,
    ] = {}

    records: list[
        dict[str, object]
    ] = []

    for row in scoring.itertuples(
        index=False
    ):
        payment_id = str(
            row.payment_id
        )

        payment_events = (
            events_by_payment.get(
                payment_id,
                empty_events,
            )
        )

        compiled = compile_support_text(
            payment_id=payment_id,
            scoring_at=row.scoring_at,
            events=payment_events,
        )

        prior_delivery = (
            delivery_by_input_hash.get(
                compiled.input_hash
            )
        )

        if prior_delivery is None:
            delivery = (
                active_service.generate(
                    compiled
                )
            )

            if (
                delivery.input_hash
                != compiled.input_hash
            ):
                raise AssertionError(
                    "Support-signal delivery "
                    "returned the wrong input hash"
                )

            delivery_by_input_hash[
                compiled.input_hash
            ] = delivery
        else:
            delivery = _reuse_delivery(
                prior_delivery,
                payment_id=payment_id,
            )

        signals = (
            delivery.signals
        )

        records.append(
            {
                "payment_id": payment_id,
                "scoring_at": (
                    compiled.scoring_at
                ),
                "intent_to_cancel": (
                    signals.intent_to_cancel
                ),
                "non_receipt_complaint": (
                    signals
                    .non_receipt_complaint
                ),
                "dissatisfaction": (
                    signals.dissatisfaction
                ),
                "input_hash": (
                    compiled.input_hash
                ),
                "delivery_mode": (
                    delivery.delivery_mode
                ),
                "provider": (
                    delivery.provider
                ),
                "model": delivery.model,
                "extraction_version": (
                    delivery
                    .extraction_version
                ),
                "cache_hit": (
                    delivery.cache_hit
                ),
                "fallback_used": (
                    delivery.fallback_used
                ),
                "fallback_reason": (
                    delivery.fallback_reason
                ),
                "observable_event_count": (
                    compiled
                    .observable_event_count
                ),
                "excluded_future_event_count": (
                    compiled
                    .excluded_future_event_count
                ),
                "excluded_injection_event_count": (
                    compiled
                    .excluded_injection_event_count
                ),
                "excluded_overflow_event_count": (
                    compiled
                    .excluded_overflow_event_count
                ),
                "pii_redaction_count": (
                    compiled
                    .pii_redaction_count
                ),
                "support_text_contract_version": (
                    SUPPORT_TEXT_CONTRACT_VERSION
                ),
                "report_version": (
                    SUPPORT_SIGNAL_REPORT_VERSION
                ),
            }
        )

    report = pd.DataFrame.from_records(
        records
    )

    expected_ids = (
        scoring["payment_id"]
        .astype(str)
        .tolist()
    )

    actual_ids = (
        report["payment_id"]
        .astype(str)
        .tolist()
    )

    if actual_ids != expected_ids:
        raise AssertionError(
            "Support-signal report is not "
            "aligned with scoring rows"
        )

    if report[
        "payment_id"
    ].duplicated().any():
        raise AssertionError(
            "Support-signal report contains "
            "duplicate payment IDs"
        )

    if (
        set(report.columns)
        & FORBIDDEN_REPORT_COLUMNS
    ):
        raise AssertionError(
            "Forbidden fields reached the "
            "support-signal report"
        )

    signal_columns = [
        "intent_to_cancel",
        "non_receipt_complaint",
        "dissatisfaction",
    ]

    for column in signal_columns:
        if report[column].isna().any():
            raise AssertionError(
                "Support-signal report "
                f"contains missing {column}"
            )

        report[column] = (
            report[column].astype(bool)
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_parquet(
        output_path,
        index=False,
    )

    return report


def main() -> None:
    report = (
        generate_support_signal_report()
    )

    unique_hashes = int(
        report["input_hash"].nunique()
    )

    live_rows = int(
        (
            report["delivery_mode"]
            == "LIVE_OPENAI"
        ).sum()
    )

    cache_rows = int(
        (
            report["delivery_mode"]
            == "CACHE"
        ).sum()
    )

    fallback_rows = int(
        report[
            "fallback_used"
        ].sum()
    )

    print(
        "Support-signal report generated"
    )
    print(
        f"Rows: {len(report):,}"
    )
    print(
        f"Unique sanitized model inputs: "
        f"{unique_hashes:,}"
    )
    print(
        f"Live OpenAI rows: {live_rows:,}"
    )
    print(
        f"Cache rows: {cache_rows:,}"
    )
    print(
        f"Fallback rows: {fallback_rows:,}"
    )

    for signal in (
        "intent_to_cancel",
        "non_receipt_complaint",
        "dissatisfaction",
    ):
        positives = int(
            report[signal].sum()
        )

        rate = (
            positives
            / len(report)
            if len(report)
            else 0.0
        )

        print(
            f"{signal}: "
            f"{positives:,} "
            f"({rate:.3%})"
        )

    print(
        "Output:",
        SUPPORT_SIGNAL_REPORT_PATH.relative_to(
            ROOT
        ),
    )


if __name__ == "__main__":
    main()