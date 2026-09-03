from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping


ACTION_PRIORITY = {
    "MONITOR": 0,
    "PREPARE_EVIDENCE": 1,
    "MANUAL_REVIEW": 2,
    "RECOMMEND_REFUND": 3,
}


@dataclass(frozen=True)
class RuleHit:
    rule_id: str
    title: str
    category: str
    severity: str
    risk_points: float
    recommended_action: str
    explanation: str
    evidence: dict[str, Any]
    hard_override: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def safe_number(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        result = float(value)

        if not math.isfinite(result):
            return default

        return result
    except (TypeError, ValueError):
        return default


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            "1",
            "yes",
            "y",
        }

    return False


def safe_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip().lower()


def highest_action(
    hits: list[RuleHit],
    hard_only: bool = False,
) -> str | None:
    candidates = [
        hit
        for hit in hits
        if not hard_only or hit.hard_override
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda hit: ACTION_PRIORITY[
            hit.recommended_action
        ],
    ).recommended_action


def evaluate_rules(
    payment: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Evaluate capture-time payment signals.

    These rules never use dispute outcomes or future information.
    Hard overrides are limited to clear merchant-side errors.
    Other rules provide supporting risk signals for the ML model.
    """

    hits: list[RuleHit] = []

    amount_paise = safe_number(
        payment.get("amount_paise")
    )
    amount_rupees = amount_paise / 100

    txns_last_1h = safe_number(
        payment.get("txns_last_1h")
    )
    txns_last_24h = safe_number(
        payment.get("txns_last_24h")
    )
    amount_last_24h_rupees = (
        safe_number(
            payment.get("amount_last_24h_paise")
        )
        / 100
    )

    account_age_days = safe_number(
        payment.get("account_age_days"),
        default=999,
    )
    prior_disputes = safe_number(
        payment.get("prior_disputes_count")
    )
    descriptor_clarity = safe_number(
        payment.get("descriptor_clarity_score"),
        default=1,
    )
    billing_distance = safe_number(
        payment.get("billing_shipping_distance_km")
    )

    device_is_new = safe_bool(
        payment.get("device_is_new")
    )
    country_matches = safe_bool(
        payment.get(
            "ip_country_matches_billing",
            True,
        )
    )
    proxy_or_vpn = safe_bool(
        payment.get("ip_is_proxy_or_vpn")
    )
    liability_shift = safe_bool(
        payment.get("threeds_liability_shift")
    )

    cvv_result = safe_text(
        payment.get("cvv_result")
    )
    threeds_status = safe_text(
        payment.get("threeds_status")
    )

    # Clear merchant error: duplicate capture.
    if safe_bool(
        payment.get("is_duplicate_payment")
    ):
        hits.append(
            RuleHit(
                rule_id="MERCHANT_DUPLICATE_PAYMENT",
                title="Possible duplicate payment",
                category="merchant_error",
                severity="critical",
                risk_points=0.45,
                recommended_action="RECOMMEND_REFUND",
                explanation=(
                    "The merchant system marked this payment "
                    "as a duplicate capture."
                ),
                evidence={
                    "is_duplicate_payment": True,
                    "amount_rupees": round(
                        amount_rupees,
                        2,
                    ),
                },
                hard_override=True,
            )
        )

    # Clear merchant error: charged after cancellation.
    if safe_bool(
        payment.get(
            "cancelled_subscription_billed"
        )
    ):
        hits.append(
            RuleHit(
                rule_id="MERCHANT_CANCELLED_SUBSCRIPTION",
                title="Subscription billed after cancellation",
                category="merchant_error",
                severity="critical",
                risk_points=0.45,
                recommended_action="RECOMMEND_REFUND",
                explanation=(
                    "The customer was billed after the "
                    "subscription had been cancelled."
                ),
                evidence={
                    "cancelled_subscription_billed": True,
                    "amount_rupees": round(
                        amount_rupees,
                        2,
                    ),
                },
                hard_override=True,
            )
        )

    cvv_failed = cvv_result in {
        "fail",
        "failed",
        "mismatch",
        "not_match",
    }

    threeds_failed = threeds_status in {
        "fail",
        "failed",
        "not_authenticated",
        "authentication_failed",
    }

    if (
        cvv_failed
        or (threeds_failed and not liability_shift)
    ):
        hits.append(
            RuleHit(
                rule_id="AUTHENTICATION_GAP",
                title="Card authentication weakness",
                category="true_fraud",
                severity="high",
                risk_points=0.24,
                recommended_action="MANUAL_REVIEW",
                explanation=(
                    "The payment contains failed card "
                    "authentication without liability shift."
                ),
                evidence={
                    "cvv_result": cvv_result,
                    "threeds_status": threeds_status,
                    "liability_shift": liability_shift,
                },
            )
        )

    if (
        device_is_new
        and (
            not country_matches
            or proxy_or_vpn
        )
    ):
        hits.append(
            RuleHit(
                rule_id="DEVICE_NETWORK_ANOMALY",
                title="New device with network anomaly",
                category="true_fraud",
                severity="high",
                risk_points=0.22,
                recommended_action="MANUAL_REVIEW",
                explanation=(
                    "A new device was combined with an IP "
                    "location mismatch or proxy/VPN signal."
                ),
                evidence={
                    "device_is_new": device_is_new,
                    "country_matches_billing": (
                        country_matches
                    ),
                    "proxy_or_vpn": proxy_or_vpn,
                },
            )
        )

    if (
        txns_last_1h >= 3
        or txns_last_24h >= 7
        or amount_last_24h_rupees >= 50_000
    ):
        hits.append(
            RuleHit(
                rule_id="PAYMENT_VELOCITY_SPIKE",
                title="Unusual payment velocity",
                category="true_fraud",
                severity="high",
                risk_points=0.20,
                recommended_action="MANUAL_REVIEW",
                explanation=(
                    "Recent transaction frequency or value "
                    "is unusually high."
                ),
                evidence={
                    "txns_last_1h": txns_last_1h,
                    "txns_last_24h": txns_last_24h,
                    "amount_last_24h_rupees": round(
                        amount_last_24h_rupees,
                        2,
                    ),
                },
            )
        )

    if prior_disputes >= 2:
        hits.append(
            RuleHit(
                rule_id="REPEAT_DISPUTE_HISTORY",
                title="Repeated prior disputes",
                category="friendly_fraud",
                severity="high",
                risk_points=0.25,
                recommended_action="PREPARE_EVIDENCE",
                explanation=(
                    "The customer had multiple disputes "
                    "before this payment was captured."
                ),
                evidence={
                    "prior_disputes_count": int(
                        prior_disputes
                    ),
                },
            )
        )

    if descriptor_clarity < 0.35:
        hits.append(
            RuleHit(
                rule_id="UNCLEAR_BILLING_DESCRIPTOR",
                title="Unclear billing descriptor",
                category="friendly_fraud",
                severity="medium",
                risk_points=0.12,
                recommended_action="PREPARE_EVIDENCE",
                explanation=(
                    "The billing descriptor may not be "
                    "recognisable to the customer."
                ),
                evidence={
                    "descriptor_clarity_score": round(
                        descriptor_clarity,
                        3,
                    ),
                },
            )
        )

    if (
        account_age_days <= 7
        and amount_rupees >= 20_000
    ):
        hits.append(
            RuleHit(
                rule_id="NEW_ACCOUNT_HIGH_VALUE",
                title="High-value payment from new account",
                category="true_fraud",
                severity="medium",
                risk_points=0.16,
                recommended_action="MANUAL_REVIEW",
                explanation=(
                    "A recently created account attempted "
                    "a high-value card payment."
                ),
                evidence={
                    "account_age_days": int(
                        account_age_days
                    ),
                    "amount_rupees": round(
                        amount_rupees,
                        2,
                    ),
                },
            )
        )

    if (
        device_is_new
        and billing_distance >= 1_000
    ):
        hits.append(
            RuleHit(
                rule_id="DEVICE_DISTANCE_ANOMALY",
                title="New device with address-distance anomaly",
                category="true_fraud",
                severity="medium",
                risk_points=0.14,
                recommended_action="MANUAL_REVIEW",
                explanation=(
                    "The payment combines a new device with "
                    "a large billing-to-shipping distance."
                ),
                evidence={
                    "device_is_new": device_is_new,
                    "billing_shipping_distance_km": round(
                        billing_distance,
                        1,
                    ),
                },
            )
        )

    total_risk_points = min(
        sum(hit.risk_points for hit in hits),
        1.0,
    )

    return {
        "triggered": bool(hits),
        "rule_count": len(hits),
        "rule_risk_score": round(
            total_risk_points,
            4,
        ),
        "risk_categories": sorted(
            {
                hit.category
                for hit in hits
            }
        ),
        "suggested_action": (
            highest_action(hits) or "MONITOR"
        ),
        "hard_override_action": highest_action(
            hits,
            hard_only=True,
        ),
        "hits": [
            hit.to_dict()
            for hit in hits
        ],
    }


if __name__ == "__main__":
    example_payment = {
        "amount_paise": 3_500_000,
        "device_is_new": True,
        "ip_country_matches_billing": False,
        "ip_is_proxy_or_vpn": True,
        "cvv_result": "failed",
        "threeds_status": "failed",
        "threeds_liability_shift": False,
        "txns_last_1h": 4,
        "txns_last_24h": 8,
        "account_age_days": 2,
        "descriptor_clarity_score": 0.22,
        "prior_disputes_count": 2,
        "billing_shipping_distance_km": 1_500,
        "is_duplicate_payment": False,
        "cancelled_subscription_billed": False,
    }

    print(
        json.dumps(
            evaluate_rules(example_payment),
            indent=2,
        )
    )