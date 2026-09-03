from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class LifecycleRuleHit:
    code: str
    category: str
    severity: str
    recommended_action: str
    rationale: str
    evidence: dict[str, Any]
    hard_override: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _value(
    record: Mapping[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    try:
        value = record.get(key, default)
    except AttributeError:
        value = getattr(record, key, default)

    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    return value


def _bool(
    record: Mapping[str, Any],
    key: str,
    default: bool = False,
) -> bool:
    value = _value(record, key, default)

    if isinstance(value, str):
        return value.strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }

    return bool(value)


def _timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None

    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None

    if pd.isna(timestamp):
        return None

    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")

    return timestamp.tz_convert("UTC")


def _result(
    hits: list[LifecycleRuleHit],
    evaluated_at: pd.Timestamp,
) -> dict[str, Any]:
    hard_override_action = next(
        (
            hit.recommended_action
            for hit in hits
            if hit.hard_override
        ),
        None,
    )

    return {
        "stage": "POST_PAYMENT",
        "evaluated_at": evaluated_at.isoformat(),
        "triggered": bool(hits),
        "triggered_count": len(hits),
        "hits": [hit.to_dict() for hit in hits],
        "hard_override_action": hard_override_action,
        "requires_human_approval": any(
            hit.recommended_action
            in {
                "MANUAL_REVIEW",
                "RECOMMEND_REFUND",
            }
            for hit in hits
        ),
        "action_executed": False,
    }


def evaluate_post_payment_rules(
    operation: Mapping[str, Any],
    *,
    as_of: datetime | pd.Timestamp | str | None = None,
) -> dict[str, Any]:
    """Evaluate lifecycle rules using facts visible by ``as_of``.

    The precomputed outcome flags in operations.parquet are deliberately
    ignored. Decisions are reconstructed from event timestamps, preventing
    future delivery or refund outcomes from leaking into an earlier decision.
    """
    evaluated_at = _timestamp(as_of)

    if evaluated_at is None:
        evaluated_at = pd.Timestamp.now(tz="UTC")

    hits: list[LifecycleRuleHit] = []

    shipment_expected = _bool(
        operation,
        "shipment_expected",
    )
    shipment_promised_at = _timestamp(
        _value(operation, "shipment_promised_at")
    )
    shipment_delivered_at = _timestamp(
        _value(operation, "shipment_delivered_at")
    )

    if (
        shipment_expected
        and shipment_promised_at is not None
        and evaluated_at > shipment_promised_at
    ):
        visible_delivery = shipment_delivered_at

        if (
            visible_delivery is not None
            and visible_delivery > evaluated_at
        ):
            visible_delivery = None

        shipment_breached = (
            visible_delivery is None
            or visible_delivery > shipment_promised_at
        )

        if shipment_breached:
            hits.append(
                LifecycleRuleHit(
                    code="SHIPMENT_SLA_BREACHED",
                    category="merchant_error",
                    severity="high",
                    recommended_action="PREPARE_EVIDENCE",
                    rationale=(
                        "The promised shipment deadline passed "
                        "without an on-time delivery."
                    ),
                    evidence={
                        "shipment_promised_at": (
                            shipment_promised_at.isoformat()
                        ),
                        "shipment_delivered_at": (
                            visible_delivery.isoformat()
                            if visible_delivery is not None
                            else None
                        ),
                        "evaluated_at": evaluated_at.isoformat(),
                    },
                )
            )

    refund_requested_at = _timestamp(
        _value(operation, "refund_requested_at")
    )
    refund_promised_by = _timestamp(
        _value(operation, "refund_promised_by")
    )
    refund_processed_at = _timestamp(
        _value(operation, "refund_processed_at")
    )

    if (
        refund_requested_at is not None
        and refund_requested_at <= evaluated_at
        and refund_promised_by is not None
        and evaluated_at > refund_promised_by
    ):
        visible_refund = refund_processed_at

        if (
            visible_refund is not None
            and visible_refund > evaluated_at
        ):
            visible_refund = None

        refund_breached = (
            visible_refund is None
            or visible_refund > refund_promised_by
        )

        if refund_breached:
            hits.append(
                LifecycleRuleHit(
                    code="REFUND_NOT_PROCESSED",
                    category="merchant_error",
                    severity="critical",
                    recommended_action="RECOMMEND_REFUND",
                    rationale=(
                        "A promised refund was not processed "
                        "before its deadline."
                    ),
                    evidence={
                        "refund_requested_at": (
                            refund_requested_at.isoformat()
                        ),
                        "refund_promised_by": (
                            refund_promised_by.isoformat()
                        ),
                        "refund_processed_at": (
                            visible_refund.isoformat()
                            if visible_refund is not None
                            else None
                        ),
                        "evaluated_at": evaluated_at.isoformat(),
                    },
                    hard_override=True,
                )
            )

    return _result(hits, evaluated_at)
