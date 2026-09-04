from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from src.razorpay_adapter import RazorpayPayment


UTC = timezone.utc
NORMALIZATION_VERSION = "razorpay-normalizer-v1"

MERCHANT_CONTEXT_FIELDS = [
    "card_network",
    "product_category",
    "is_digital_good",
    "descriptor_clarity_score",
    "phone_verified",
    "email_verified",
    "account_age_days",
    "has_prior_order",
    "total_prior_orders",
    "prior_disputes_count",
    "days_since_last_order",
    "txns_last_1h",
    "txns_last_24h",
    "txns_last_7d",
    "amount_last_24h_paise",
    "device_is_new",
    "ip_country_matches_billing",
    "ip_is_proxy_or_vpn",
    "cvv_result",
    "threeds_status",
    "threeds_liability_shift",
    "billing_shipping_distance_km",
    "is_duplicate_payment",
    "cancelled_subscription_billed",
]


class RazorpayNormalizationError(ValueError):
    pass


class RazorpayMerchantRiskContext(BaseModel):
    """Capture-time merchant facts; no identity or outcome fields."""

    model_config = ConfigDict(
        extra="forbid",
    )

    feature_as_of: datetime

    card_network: Literal[
        "Visa",
        "Mastercard",
    ] = "Visa"

    product_category: Literal[
        "fashion",
        "electronics",
        "travel",
        "food",
        "subscription",
        "gaming",
        "education",
    ] = "electronics"

    is_digital_good: bool = False

    descriptor_clarity_score: float = Field(
        default=0.80,
        ge=0,
        le=1,
    )

    phone_verified: bool = True
    email_verified: bool = True

    account_age_days: int = Field(
        default=365,
        ge=0,
        le=10_000,
    )

    has_prior_order: int = Field(
        default=1,
        ge=0,
        le=1,
    )

    total_prior_orders: int = Field(
        default=5,
        ge=0,
        le=100_000,
    )

    prior_disputes_count: int = Field(
        default=0,
        ge=0,
        le=10_000,
    )

    days_since_last_order: float = Field(
        default=30,
        ge=0,
        le=999,
    )

    txns_last_1h: int = Field(
        default=0,
        ge=0,
    )

    txns_last_24h: int = Field(
        default=0,
        ge=0,
    )

    txns_last_7d: int = Field(
        default=1,
        ge=0,
    )

    amount_last_24h_paise: int = Field(
        default=0,
        ge=0,
    )

    device_is_new: bool = False
    ip_country_matches_billing: bool = True
    ip_is_proxy_or_vpn: bool = False

    cvv_result: Literal[
        "match",
        "no_match",
        "not_provided",
    ] = "match"

    threeds_status: Literal[
        "authenticated",
        "attempted",
        "not_authenticated",
    ] = "authenticated"

    threeds_liability_shift: bool = True

    billing_shipping_distance_km: float = Field(
        default=10,
        ge=0,
        le=50_000,
    )

    is_duplicate_payment: bool = False
    cancelled_subscription_billed: bool = False

    @field_validator("feature_as_of")
    @classmethod
    def require_timezone(
        cls,
        value: datetime,
    ) -> datetime:
        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "feature_as_of must include a timezone."
            )

        return value.astimezone(UTC)


class CaptureTimeScoringPayload(BaseModel):
    """Exact safe payload accepted by the existing scorer."""

    model_config = ConfigDict(
        extra="forbid",
    )

    created_at: datetime

    amount_paise: int = Field(
        ge=100,
        le=100_000_000,
    )

    card_network: Literal[
        "Visa",
        "Mastercard",
    ]

    product_category: Literal[
        "fashion",
        "electronics",
        "travel",
        "food",
        "subscription",
        "gaming",
        "education",
    ]

    is_digital_good: bool
    descriptor_clarity_score: float
    phone_verified: bool
    email_verified: bool
    account_age_days: int
    has_prior_order: int
    total_prior_orders: int
    prior_disputes_count: int
    days_since_last_order: float
    txns_last_1h: int
    txns_last_24h: int
    txns_last_7d: int
    amount_last_24h_paise: int
    device_is_new: bool
    ip_country_matches_billing: bool
    ip_is_proxy_or_vpn: bool

    cvv_result: Literal[
        "match",
        "no_match",
        "not_provided",
    ]

    threeds_status: Literal[
        "authenticated",
        "attempted",
        "not_authenticated",
    ]

    threeds_liability_shift: bool
    billing_shipping_distance_km: float
    is_duplicate_payment: bool
    cancelled_subscription_billed: bool


class NormalizedRazorpayPayment(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    normalization_version: Literal[
        "razorpay-normalizer-v1"
    ] = NORMALIZATION_VERSION

    source: Literal[
        "RAZORPAY_TEST_MODE"
    ] = "RAZORPAY_TEST_MODE"

    razorpay_payment_id: str
    razorpay_order_id: str
    provider_status: Literal[
        "authorized",
        "captured",
    ]

    feature_as_of: datetime
    scoring_payload: CaptureTimeScoringPayload
    merchant_context_fields: list[str]

    synthetic_evaluation_affected: Literal[
        False
    ] = False

    disclosure: Literal[
        "Razorpay Test Mode flow; not part of held-out model evaluation."
    ] = (
        "Razorpay Test Mode flow; not part of held-out "
        "model evaluation."
    )


def normalize_razorpay_payment(
    payment: RazorpayPayment,
    context: RazorpayMerchantRiskContext,
) -> NormalizedRazorpayPayment:
    if payment.currency.upper() != "INR":
        raise RazorpayNormalizationError(
            "Only INR payments are supported."
        )

    if payment.method != "card":
        raise RazorpayNormalizationError(
            "The current model is validated only for card "
            "payments."
        )

    if payment.status not in {
        "authorized",
        "captured",
    }:
        raise RazorpayNormalizationError(
            "Only authorized or captured payments can be "
            "scored."
        )

    if not payment.order_id:
        raise RazorpayNormalizationError(
            "The Razorpay payment must belong to an order."
        )

    payment_created_at = datetime.fromtimestamp(
        payment.created_at,
        tz=UTC,
    )

    if context.feature_as_of > payment_created_at:
        raise RazorpayNormalizationError(
            "Merchant context was observed after the payment "
            "scoring timestamp."
        )

    context_values = context.model_dump(
        exclude={"feature_as_of"},
    )

    scoring_payload = CaptureTimeScoringPayload(
        created_at=payment_created_at,
        amount_paise=payment.amount,
        **context_values,
    )

    return NormalizedRazorpayPayment(
        razorpay_payment_id=payment.id,
        razorpay_order_id=payment.order_id,
        provider_status=payment.status,
        feature_as_of=context.feature_as_of,
        scoring_payload=scoring_payload,
        merchant_context_fields=(
            MERCHANT_CONTEXT_FIELDS.copy()
        ),
    )