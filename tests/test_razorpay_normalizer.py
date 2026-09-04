from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from src.razorpay_adapter import RazorpayPayment
from src.razorpay_normalizer import (
    MERCHANT_CONTEXT_FIELDS,
    RazorpayMerchantRiskContext,
    RazorpayNormalizationError,
    normalize_razorpay_payment,
)


UTC = timezone.utc

PAYMENT_TIME = datetime(
    2026,
    9,
    4,
    12,
    0,
    tzinfo=UTC,
)


def _payment(
    *,
    method: str = "card",
    status: str = "captured",
    currency: str = "INR",
    order_id: str | None = "order_Test123",
) -> RazorpayPayment:
    return RazorpayPayment.model_validate(
        {
            "id": "pay_Test123",
            "entity": "payment",
            "amount": 250000,
            "currency": currency,
            "status": status,
            "order_id": order_id,
            "method": method,
            "captured": status == "captured",
            "description": "Test payment",
            "created_at": int(
                PAYMENT_TIME.timestamp()
            ),
            "email": "forbidden@example.com",
            "contact": "+919999999999",
            "customer_id": "cust_forbidden",
            "amount_refunded": 0,
            "refund_status": None,
        }
    )


def _context(
    **overrides,
) -> RazorpayMerchantRiskContext:
    values = {
        "feature_as_of": (
            PAYMENT_TIME
            - timedelta(seconds=5)
        ),
        "card_network": "Visa",
        "product_category": "electronics",
        "is_digital_good": False,
        "descriptor_clarity_score": 0.82,
        "phone_verified": True,
        "email_verified": True,
        "account_age_days": 420,
        "has_prior_order": 1,
        "total_prior_orders": 8,
        "prior_disputes_count": 0,
        "days_since_last_order": 20,
        "txns_last_1h": 0,
        "txns_last_24h": 1,
        "txns_last_7d": 2,
        "amount_last_24h_paise": 250000,
        "device_is_new": False,
        "ip_country_matches_billing": True,
        "ip_is_proxy_or_vpn": False,
        "cvv_result": "match",
        "threeds_status": "authenticated",
        "threeds_liability_shift": True,
        "billing_shipping_distance_km": 12,
        "is_duplicate_payment": False,
        "cancelled_subscription_billed": False,
    }

    values.update(overrides)

    return RazorpayMerchantRiskContext.model_validate(
        values
    )


def test_normalization_is_deterministic() -> None:
    first = normalize_razorpay_payment(
        _payment(),
        _context(),
    )

    second = normalize_razorpay_payment(
        _payment(),
        _context(),
    )

    assert first.model_dump() == second.model_dump()


def test_provider_controls_amount_and_timestamp() -> None:
    result = normalize_razorpay_payment(
        _payment(),
        _context(),
    )

    payload = result.scoring_payload

    assert payload.amount_paise == 250000
    assert payload.created_at == PAYMENT_TIME
    assert result.razorpay_payment_id == "pay_Test123"
    assert result.razorpay_order_id == "order_Test123"


def test_model_boundary_contains_exact_safe_fields() -> None:
    result = normalize_razorpay_payment(
        _payment(),
        _context(),
    )

    payload = result.scoring_payload.model_dump()

    expected = {
        "created_at",
        "amount_paise",
        *MERCHANT_CONTEXT_FIELDS,
    }

    assert set(payload) == expected


def test_identity_and_outcomes_never_cross_model_boundary() -> None:
    result = normalize_razorpay_payment(
        _payment(),
        _context(),
    )

    payload = result.scoring_payload.model_dump()

    forbidden = {
        "payment_id",
        "order_id",
        "customer_id",
        "email",
        "contact",
        "chargeback_within_120d",
        "target",
        "true_fraud",
        "dispute_status",
        "dispute_reason",
        "refund_status",
        "amount_refunded",
        "provider_status",
        "method",
    }

    assert forbidden.isdisjoint(payload)

    serialized = str(payload)

    assert "forbidden@example.com" not in serialized
    assert "+919999999999" not in serialized
    assert "cust_forbidden" not in serialized


def test_context_rejects_unknown_sensitive_fields() -> None:
    values = _context().model_dump()
    values["customer_id"] = "cust_forbidden"

    with pytest.raises(ValidationError):
        RazorpayMerchantRiskContext.model_validate(
            values
        )


def test_context_requires_timezone() -> None:
    values = _context().model_dump()

    values["feature_as_of"] = datetime(
        2026,
        9,
        4,
        11,
        59,
    )

    with pytest.raises(ValidationError):
        RazorpayMerchantRiskContext.model_validate(
            values
        )


def test_future_context_is_rejected() -> None:
    context = _context(
        feature_as_of=(
            PAYMENT_TIME
            + timedelta(seconds=1)
        )
    )

    with pytest.raises(
        RazorpayNormalizationError,
        match="after the payment",
    ):
        normalize_razorpay_payment(
            _payment(),
            context,
        )


@pytest.mark.parametrize(
    "method",
    [
        "upi",
        "wallet",
        "netbanking",
    ],
)
def test_non_card_payment_is_rejected(
    method: str,
) -> None:
    with pytest.raises(
        RazorpayNormalizationError,
        match="card payments",
    ):
        normalize_razorpay_payment(
            _payment(method=method),
            _context(),
        )


@pytest.mark.parametrize(
    "status",
    [
        "created",
        "failed",
        "refunded",
    ],
)
def test_unsafe_payment_state_is_rejected(
    status: str,
) -> None:
    with pytest.raises(
        RazorpayNormalizationError,
        match="authorized or captured",
    ):
        normalize_razorpay_payment(
            _payment(status=status),
            _context(),
        )


def test_non_inr_payment_is_rejected() -> None:
    with pytest.raises(
        RazorpayNormalizationError,
        match="INR",
    ):
        normalize_razorpay_payment(
            _payment(currency="USD"),
            _context(),
        )


def test_payment_without_order_is_rejected() -> None:
    with pytest.raises(
        RazorpayNormalizationError,
        match="belong to an order",
    ):
        normalize_razorpay_payment(
            _payment(order_id=None),
            _context(),
        )


def test_flow_is_disclosed_as_separate_from_evaluation() -> None:
    result = normalize_razorpay_payment(
        _payment(),
        _context(),
    )

    assert result.synthetic_evaluation_affected is False

    assert result.disclosure == (
        "Razorpay Test Mode flow; not part of "
        "held-out model evaluation."
    )