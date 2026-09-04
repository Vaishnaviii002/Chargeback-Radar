from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Mapping

import pandas as pd

from src.evidence_contract import (
    EvidenceCase,
    EvidenceFact,
    FactSource,
)
from src.post_payment_rules import evaluate_post_payment_rules
from src.rules import evaluate_rules


ACTIONS = {
    "MONITOR",
    "PREPARE_EVIDENCE",
    "MANUAL_REVIEW",
    "RECOMMEND_REFUND",
}

# These fields may exist in held-out data, but must never reach the model.
FORBIDDEN_INPUT_FIELDS = {
    "customer_id",
    "chargeback_within_120d",
    "label",
    "chargeback_family",
    "reason_code",
    "dispute_created_at",
    "dispute_status",
    "respond_by",
    "shipment_sla_breached",
    "promised_refund_not_processed",
}


class EvidenceFactError(ValueError):
    pass


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


def _boolean_text(value: Any) -> str:
    if isinstance(value, str):
        result = value.strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }
    else:
        result = bool(value)

    return "Yes" if result else "No"


def _risk_band(probability: float) -> str:
    if probability < 0.01:
        return "LOW"
    if probability < 0.05:
        return "MEDIUM"
    if probability < 0.15:
        return "HIGH"
    return "CRITICAL"


def _action(value: Any, field_name: str) -> str:
    action = str(value)

    if action not in ACTIONS:
        raise EvidenceFactError(
            f"{field_name} must be one of {sorted(ACTIONS)}."
        )

    return action


def _add_fact(
    facts: list[EvidenceFact],
    *,
    fact_id: str,
    source: FactSource,
    label: str,
    value: Any,
    observed_at: pd.Timestamp | datetime | None = None,
) -> None:
    if value is None:
        return

    timestamp = _timestamp(observed_at)
    facts.append(
        EvidenceFact(
            fact_id=fact_id,
            source=source,
            label=label,
            value=str(value),
            observed_at=(
                timestamp.to_pydatetime()
                if timestamp is not None
                else None
            ),
        )
    )


def _capture_time_facts(
    payment: Mapping[str, Any],
    created_at: pd.Timestamp,
) -> list[EvidenceFact]:
    facts: list[EvidenceFact] = []

    _add_fact(
        facts,
        fact_id="PAYMENT_CREATED_AT",
        source=FactSource.PAYMENT,
        label="Payment created at",
        value=created_at.isoformat(),
        observed_at=created_at,
    )

    amount_paise = _value(payment, "amount_paise")
    if amount_paise is not None:
        amount_rupees = float(amount_paise) / 100
        _add_fact(
            facts,
            fact_id="PAYMENT_AMOUNT",
            source=FactSource.PAYMENT,
            label="Payment amount",
            value=f"INR {amount_rupees:,.2f}",
            observed_at=created_at,
        )

    text_fields = [
        ("card_network", "CARD_NETWORK", "Card network"),
        (
            "product_category",
            "PRODUCT_CATEGORY",
            "Product category",
        ),
        ("cvv_result", "CVV_RESULT", "CVV result"),
        (
            "threeds_status",
            "THREEDS_STATUS",
            "3DS status",
        ),
    ]

    for field, fact_id, label in text_fields:
        _add_fact(
            facts,
            fact_id=fact_id,
            source=FactSource.PAYMENT,
            label=label,
            value=_value(payment, field),
            observed_at=created_at,
        )

    boolean_fields = [
        ("is_digital_good", "DIGITAL_GOOD", "Digital good"),
        ("phone_verified", "PHONE_VERIFIED", "Phone verified"),
        ("email_verified", "EMAIL_VERIFIED", "Email verified"),
        ("device_is_new", "DEVICE_NEW", "New device"),
        (
            "ip_country_matches_billing",
            "IP_COUNTRY_MATCH",
            "IP country matches billing",
        ),
        (
            "ip_is_proxy_or_vpn",
            "IP_PROXY_VPN",
            "Proxy or VPN detected",
        ),
        (
            "threeds_liability_shift",
            "THREEDS_LIABILITY_SHIFT",
            "3DS liability shift",
        ),
        (
            "is_duplicate_payment",
            "DUPLICATE_PAYMENT",
            "Duplicate payment detected",
        ),
        (
            "cancelled_subscription_billed",
            "CANCELLED_SUBSCRIPTION_BILLED",
            "Cancelled subscription billed",
        ),
    ]

    for field, fact_id, label in boolean_fields:
        value = _value(payment, field)
        if value is not None:
            _add_fact(
                facts,
                fact_id=fact_id,
                source=FactSource.PAYMENT,
                label=label,
                value=_boolean_text(value),
                observed_at=created_at,
            )

    numeric_fields = [
        (
            "descriptor_clarity_score",
            "DESCRIPTOR_CLARITY",
            "Descriptor clarity score",
        ),
        ("account_age_days", "ACCOUNT_AGE_DAYS", "Account age in days"),
        ("total_prior_orders", "PRIOR_ORDERS", "Prior orders"),
        (
            "prior_disputes_count",
            "PRIOR_DISPUTES",
            "Prior disputes known at payment time",
        ),
        ("txns_last_1h", "VELOCITY_1H", "Transactions in prior hour"),
        (
            "txns_last_24h",
            "VELOCITY_24H",
            "Transactions in prior 24 hours",
        ),
        ("txns_last_7d", "VELOCITY_7D", "Transactions in prior 7 days"),
        (
            "billing_shipping_distance_km",
            "BILLING_SHIPPING_DISTANCE",
            "Billing to shipping distance in kilometres",
        ),
    ]

    for field, fact_id, label in numeric_fields:
        _add_fact(
            facts,
            fact_id=fact_id,
            source=FactSource.PAYMENT,
            label=label,
            value=_value(payment, field),
            observed_at=created_at,
        )

    prior_amount = _value(payment, "amount_last_24h_paise")
    if prior_amount is not None:
        _add_fact(
            facts,
            fact_id="AMOUNT_PRIOR_24H",
            source=FactSource.PAYMENT,
            label="Payment amount in prior 24 hours",
            value=f"INR {float(prior_amount) / 100:,.2f}",
            observed_at=created_at,
        )

    return facts


def _operation_facts(
    operation: Mapping[str, Any],
    *,
    created_at: pd.Timestamp,
    as_of: pd.Timestamp,
) -> list[EvidenceFact]:
    facts: list[EvidenceFact] = []

    shipment_expected = _value(operation, "shipment_expected")
    if shipment_expected is not None:
        _add_fact(
            facts,
            fact_id="SHIPMENT_EXPECTED",
            source=FactSource.ORDER,
            label="Physical shipment expected",
            value=_boolean_text(shipment_expected),
            observed_at=created_at,
        )

    shipment_promised_at = _timestamp(
        _value(operation, "shipment_promised_at")
    )
    if shipment_promised_at is not None:
        _add_fact(
            facts,
            fact_id="SHIPMENT_PROMISED_AT",
            source=FactSource.SHIPMENT,
            label="Promised shipment deadline",
            value=shipment_promised_at.isoformat(),
            # A promise can point to the future but is known at purchase.
            observed_at=created_at,
        )

    shipment_delivered_at = _timestamp(
        _value(operation, "shipment_delivered_at")
    )
    if (
        shipment_delivered_at is not None
        and shipment_delivered_at <= as_of
    ):
        _add_fact(
            facts,
            fact_id="SHIPMENT_DELIVERED_AT",
            source=FactSource.SHIPMENT,
            label="Shipment delivered at",
            value=shipment_delivered_at.isoformat(),
            observed_at=shipment_delivered_at,
        )

    refund_requested_at = _timestamp(
        _value(operation, "refund_requested_at")
    )
    refund_is_visible = (
        refund_requested_at is not None
        and refund_requested_at <= as_of
    )

    if refund_is_visible:
        _add_fact(
            facts,
            fact_id="REFUND_REQUESTED_AT",
            source=FactSource.REFUND,
            label="Refund requested at",
            value=refund_requested_at.isoformat(),
            observed_at=refund_requested_at,
        )

        refund_promised_by = _timestamp(
            _value(operation, "refund_promised_by")
        )
        if refund_promised_by is not None:
            _add_fact(
                facts,
                fact_id="REFUND_PROMISED_BY",
                source=FactSource.REFUND,
                label="Refund promised by",
                value=refund_promised_by.isoformat(),
                # The deadline becomes known with the refund request.
                observed_at=refund_requested_at,
            )

        refund_processed_at = _timestamp(
            _value(operation, "refund_processed_at")
        )
        if (
            refund_processed_at is not None
            and refund_processed_at <= as_of
        ):
            _add_fact(
                facts,
                fact_id="REFUND_PROCESSED_AT",
                source=FactSource.REFUND,
                label="Refund processed at",
                value=refund_processed_at.isoformat(),
                observed_at=refund_processed_at,
            )

    return facts


def _first_present(
    record: Mapping[str, Any],
    keys: tuple[str, ...],
) -> Any:
    for key in keys:
        value = record.get(key)

        if value is not None and str(value).strip():
            return value

    return None


def _rule_hit_values(
    rule_result: Mapping[str, Any],
) -> list[Any]:
    for key in (
        "hits",
        "triggered_rules",
        "rules_triggered",
        "rules",
    ):
        values = rule_result.get(key)

        if isinstance(values, (list, tuple)) and values:
            return list(values)

    reasons = rule_result.get("reasons")

    if isinstance(reasons, (list, tuple)):
        return list(reasons)

    return []


def _canonical_rule_code(
    raw_code: Any,
    searchable_text: str,
    index: int,
) -> str:
    combined = f"{raw_code or ''} {searchable_text}".upper()

    if "DUPLICATE" in combined:
        return "MERCHANT_DUPLICATE_PAYMENT"

    if "CANCEL" in combined and "SUBSCRIPTION" in combined:
        return "MERCHANT_CANCELLED_SUBSCRIPTION"

    if "AUTHENTICATION" in combined or "LIABILITY SHIFT" in combined:
        return "AUTHENTICATION_GAP"

    if "VELOCITY" in combined:
        return "PAYMENT_VELOCITY_SPIKE"

    if "DESCRIPTOR" in combined:
        return "UNCLEAR_BILLING_DESCRIPTOR"

    if "DISPUTE" in combined and "HISTORY" in combined:
        return "REPEAT_DISPUTE_HISTORY"

    if "NEW ACCOUNT" in combined:
        return "NEW_ACCOUNT_HIGH_VALUE"

    if "DEVICE" in combined and (
        "NETWORK" in combined
        or "PROXY" in combined
        or "CVV" in combined
    ):
        return "DEVICE_NETWORK_ANOMALY"

    code = re.sub(
        r"[^A-Z0-9]+",
        "_",
        str(raw_code or f"RULE_{index}").upper(),
    ).strip("_")

    if not code:
        code = f"RULE_{index}"

    return code[:50].rstrip("_")


def _normalized_rule_hits(
    rule_result: Mapping[str, Any],
) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []

    for index, hit in enumerate(
        _rule_hit_values(rule_result),
        start=1,
    ):
        if isinstance(hit, Mapping):
            raw_code = _first_present(
                hit,
                (
                    "code",
                    "rule_code",
                    "rule_id",
                    "rule_name",
                    "name",
                    "id",
                    "rule",
                    "title",
                ),
            )

            raw_rationale = _first_present(
                hit,
                (
                    "rationale",
                    "reason",
                    "message",
                    "description",
                    "detail",
                ),
            )

            searchable_text = " ".join(
                str(value)
                for value in hit.values()
                if value is not None
            )
        else:
            raw_code = hit
            raw_rationale = hit
            searchable_text = str(hit)

        code = _canonical_rule_code(
            raw_code,
            searchable_text,
            index,
        )

        rationale = str(
            raw_rationale
            or f"Deterministic rule {code} was triggered."
        )

        normalized.append((code, rationale))

    return normalized


def _rule_facts(
    rule_result: Mapping[str, Any],
    *,
    prefix: str,
    source: FactSource,
    observed_at: pd.Timestamp,
) -> list[EvidenceFact]:
    facts: list[EvidenceFact] = []
    used_fact_ids: set[str] = set()

    for index, (code, rationale) in enumerate(
        _normalized_rule_hits(rule_result),
        start=1,
    ):
        base_fact_id = f"{prefix}_{code}"[:63].rstrip("_")
        fact_id = base_fact_id

        if fact_id in used_fact_ids:
            suffix = f"_{index}"
            fact_id = (
                base_fact_id[: 63 - len(suffix)]
                + suffix
            )

        used_fact_ids.add(fact_id)

        _add_fact(
            facts,
            fact_id=fact_id,
            source=source,
            label=f"Triggered rule: {code}",
            value=rationale,
            observed_at=observed_at,
        )

    return facts

def build_evidence_case(
    payment: Mapping[str, Any],
    *,
    operation: Mapping[str, Any] | None,
    as_of: datetime | pd.Timestamp | str,
) -> EvidenceCase:
    """Build the only fact envelope the drafting model may receive.

    This function uses an explicit allowlist and reconstructs lifecycle rules
    as of a point in time. It never serializes labels, dispute outcomes, raw
    customer identifiers, or future fulfilment/refund events.
    """
    payment_id = str(_value(payment, "payment_id", "")).strip()
    if not payment_id:
        raise EvidenceFactError("payment_id is required.")

    evaluated_at = _timestamp(as_of)
    if evaluated_at is None:
        raise EvidenceFactError("as_of must be a valid timestamp.")

    created_at = _timestamp(_value(payment, "created_at"))
    if created_at is None:
        raise EvidenceFactError("created_at is required.")
    if created_at > evaluated_at:
        raise EvidenceFactError("as_of cannot be before the payment.")

    if operation is not None:
        operation_payment_id = str(
            _value(operation, "payment_id", payment_id)
        )
        if operation_payment_id != payment_id:
            raise EvidenceFactError(
                "The operation does not belong to this payment."
            )
    else:
        operation = {}

    probability_value = _value(payment, "calibrated_probability")
    if probability_value is None:
        raise EvidenceFactError("calibrated_probability is required.")

    probability = float(probability_value)
    if not 0 <= probability <= 1:
        raise EvidenceFactError(
            "calibrated_probability must be between 0 and 1."
        )

    policy_action = _action(
        _value(payment, "recommended_action"),
        "recommended_action",
    )

    capture_rules = evaluate_rules(payment)
    lifecycle_rules = evaluate_post_payment_rules(
        operation,
        as_of=evaluated_at,
    )

    deterministic_action = (
        lifecycle_rules.get("hard_override_action")
        or capture_rules.get("hard_override_action")
        or policy_action
    )
    deterministic_action = _action(
        deterministic_action,
        "deterministic_recommended_action",
    )

    risk_band = _risk_band(probability)
    facts = _capture_time_facts(payment, created_at)

    _add_fact(
        facts,
        fact_id="MODEL_CALIBRATED_PROBABILITY",
        source=FactSource.MODEL,
        label="Calibrated chargeback probability",
        value=f"{probability:.2%}",
        observed_at=created_at,
    )
    _add_fact(
        facts,
        fact_id="MODEL_RISK_BAND",
        source=FactSource.MODEL,
        label="Risk band",
        value=risk_band,
        observed_at=created_at,
    )
    _add_fact(
        facts,
        fact_id="POLICY_RECOMMENDED_ACTION",
        source=FactSource.POLICY,
        label="Lowest expected-cost policy action",
        value=policy_action,
        observed_at=created_at,
    )

    facts.extend(
        _operation_facts(
            operation,
            created_at=created_at,
            as_of=evaluated_at,
        )
    )
    facts.extend(
        _rule_facts(
            capture_rules,
            prefix="CAPTURE",
            source=FactSource.CAPTURE_RULE,
            observed_at=created_at,
        )
    )
    facts.extend(
        _rule_facts(
            lifecycle_rules,
            prefix="LIFECYCLE",
            source=FactSource.LIFECYCLE_RULE,
            observed_at=evaluated_at,
        )
    )

    triggered_rule_codes = list(
    dict.fromkeys(
        code
        for result in (capture_rules, lifecycle_rules)
        for code, _ in _normalized_rule_hits(result)
    )
)

    return EvidenceCase(
        payment_id=payment_id,
        as_of=evaluated_at.to_pydatetime(),
        deterministic_recommended_action=deterministic_action,
        calibrated_probability=probability,
        risk_band=risk_band,
        triggered_rule_codes=triggered_rule_codes,
        facts=facts,
    )
