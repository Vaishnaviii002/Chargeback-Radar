from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.razorpay_checkout import (
    RazorpayCheckoutVerificationResult,
)
from src.razorpay_normalizer import (
    RazorpayMerchantRiskContext,
)
from src.razorpay_service import (
    RazorpayPaymentRiskResult,
    RazorpayRiskService,
    RazorpayServiceAuthenticationError,
)
from src.risk_scoring import RiskScoreResult


UTC = timezone.utc
ORDER_ID = "order_TestOrder12345"
PAYMENT_ID = "pay_TestPayment12345"
SIGNATURE = "a" * 64


def make_context() -> RazorpayMerchantRiskContext:
    return RazorpayMerchantRiskContext(
        feature_as_of=datetime(
            2026,
            9,
            5,
            0,
            0,
            tzinfo=UTC,
        )
    )


def make_verification(
    *,
    order_id: str = ORDER_ID,
    payment_id: str = PAYMENT_ID,
) -> RazorpayCheckoutVerificationResult:
    return RazorpayCheckoutVerificationResult(
        verified=True,
        razorpay_order_id=order_id,
        razorpay_payment_id=payment_id,
        algorithm="HMAC_SHA256",
        source="RAZORPAY_TEST_MODE",
        synthetic_evaluation_affected=False,
        financial_action_executed=False,
    )


def make_risk_result(
    *,
    order_id: str = ORDER_ID,
    payment_id: str = PAYMENT_ID,
) -> RazorpayPaymentRiskResult:
    score = RiskScoreResult.model_construct()

    return RazorpayPaymentRiskResult.model_construct(
        delivery_mode="CALCULATED",
        razorpay_payment_id=payment_id,
        razorpay_order_id=order_id,
        provider_status="authorized",
        provider_method="card",
        normalization_version=(
            "razorpay-normalizer-v1"
        ),
        score=score,
        source="RAZORPAY_TEST_MODE",
        synthetic_evaluation_affected=False,
        human_approval_required=True,
        financial_action_executed=False,
        disclosure=(
            "Razorpay Test Mode payment scored "
            "separately from the synthetic held-out "
            "evaluation. No refund, dispute or "
            "customer message was executed."
        ),
    )


def make_service() -> RazorpayRiskService:
    return RazorpayRiskService(
        adapter=SimpleNamespace(),
        store=SimpleNamespace(),
        scorer=lambda _: None,
    )


def test_verification_happens_before_scoring() -> None:
    service = make_service()
    call_order: list[str] = []

    def verify(**kwargs: object):
        call_order.append("verify")
        return make_verification()

    def score(
        payment_id: str,
        context: RazorpayMerchantRiskContext,
    ):
        call_order.append("score")
        return make_risk_result()

    service.verify_checkout_payment = Mock(
        side_effect=verify
    )

    service.score_payment = Mock(
        side_effect=score
    )

    result = service.verify_and_score_checkout(
        server_order_id=ORDER_ID,
        razorpay_payment_id=PAYMENT_ID,
        razorpay_signature=SIGNATURE,
        context=make_context(),
    )

    assert call_order == [
        "verify",
        "score",
    ]

    assert result.verification.verified is True

    assert (
        result.risk_result.razorpay_payment_id
        == PAYMENT_ID
    )

    assert (
        result.signature_required_before_scoring
        is True
    )

    assert (
        result.synthetic_evaluation_affected
        is False
    )

    assert (
        result.financial_action_executed
        is False
    )


def test_scored_order_must_match_verified_order() -> None:
    service = make_service()

    service.verify_checkout_payment = Mock(
        return_value=make_verification()
    )

    service.score_payment = Mock(
        return_value=make_risk_result(
            order_id="order_DifferentOrder123"
        )
    )

    with pytest.raises(
        RazorpayServiceAuthenticationError,
        match="order",
    ):
        service.verify_and_score_checkout(
            server_order_id=ORDER_ID,
            razorpay_payment_id=PAYMENT_ID,
            razorpay_signature=SIGNATURE,
            context=make_context(),
        )


def test_scored_payment_must_match_verified_payment() -> None:
    service = make_service()

    service.verify_checkout_payment = Mock(
        return_value=make_verification()
    )

    service.score_payment = Mock(
        return_value=make_risk_result(
            payment_id=(
                "pay_DifferentPayment123"
            )
        )
    )

    with pytest.raises(
        RazorpayServiceAuthenticationError,
        match="identity",
    ):
        service.verify_and_score_checkout(
            server_order_id=ORDER_ID,
            razorpay_payment_id=PAYMENT_ID,
            razorpay_signature=SIGNATURE,
            context=make_context(),
        )


def test_verification_failure_prevents_scoring() -> None:
    service = make_service()

    service.verify_checkout_payment = Mock(
        side_effect=(
            RazorpayServiceAuthenticationError(
                "Invalid signature"
            )
        )
    )

    service.score_payment = Mock()

    with pytest.raises(
        RazorpayServiceAuthenticationError
    ):
        service.verify_and_score_checkout(
            server_order_id=ORDER_ID,
            razorpay_payment_id=PAYMENT_ID,
            razorpay_signature=SIGNATURE,
            context=make_context(),
        )

    service.score_payment.assert_not_called()