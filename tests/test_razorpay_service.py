from datetime import datetime, timedelta, timezone

import pytest
from pydantic import SecretStr, ValidationError

from src.razorpay_adapter import (
    RazorpayOrder,
    RazorpayPayment,
    RazorpayUnavailableError,
    RazorpayConfig,
)
from src.razorpay_normalizer import (
    RazorpayMerchantRiskContext,
)
from src.razorpay_service import (
    RazorpayOperationFailedError,
    RazorpayOrderIntent,
    RazorpayRiskService,
    RazorpayServiceUnavailableError,
)
from src.razorpay_store import (
    RazorpayIdempotencyConflictError,
    RazorpayStore,
)
from src.risk_scoring import RiskScoreResult


UTC = timezone.utc

PAYMENT_TIME = datetime(
    2026,
    9,
    4,
    12,
    0,
    tzinfo=UTC,
)


class FakeAdapter:
    def __init__(self) -> None:
        self.config = RazorpayConfig(
            enabled=True,
            test_mode=True,
            key_id="rzp_test_public123",
            key_secret=SecretStr(
                "never_expose_this_secret"
            ),
        )

        self.create_calls = 0
        self.fetch_calls = 0
        self.fail_create = False

    def create_order(
        self,
        request,
    ) -> RazorpayOrder:
        self.create_calls += 1

        if self.fail_create:
            raise RazorpayUnavailableError(
                "Provider detail"
            )

        return RazorpayOrder(
            id="order_Test123",
            entity="order",
            amount=request.amount,
            amount_paid=0,
            amount_due=request.amount,
            currency=request.currency,
            receipt=request.receipt,
            status="created",
            attempts=0,
            notes=request.notes,
            created_at=int(
                PAYMENT_TIME.timestamp()
            ),
        )

    def fetch_payment(
        self,
        payment_id: str,
    ) -> RazorpayPayment:
        self.fetch_calls += 1

        return RazorpayPayment.model_validate(
            {
                "id": payment_id,
                "entity": "payment",
                "amount": 250000,
                "currency": "INR",
                "status": "captured",
                "order_id": "order_Test123",
                "method": "card",
                "captured": True,
                "created_at": int(
                    PAYMENT_TIME.timestamp()
                ),
                "email": (
                    "must-not-survive@example.com"
                ),
                "contact": "+919999999999",
                "customer_id": "cust_forbidden",
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


def _score_result() -> RiskScoreResult:
    return RiskScoreResult(
        raw_probability=0.04,
        calibrated_probability=0.02,
        risk_percentage=2.0,
        model_recommended_action="MONITOR",
        recommended_action="MONITOR",
        decision_source="COST_OPTIMIZED_MODEL",
        rules={
            "triggered_rules": [],
            "hard_override_action": None,
        },
        risk_band="MEDIUM",
        expected_costs_rupees={
            "MONITOR": 100.0,
            "PREPARE_EVIDENCE": 120.0,
            "MANUAL_REVIEW": 180.0,
            "RECOMMEND_REFUND": 2500.0,
        },
        model_version="0.1.0",
        calibration_method="isotonic",
        requires_human_approval=False,
        action_executed=False,
    )


def _service(
    tmp_path,
    *,
    adapter: FakeAdapter | None = None,
    scorer=None,
):
    selected_adapter = (
        adapter or FakeAdapter()
    )

    calls = {"score": 0}

    def default_scorer(payload):
        calls["score"] += 1
        return _score_result()

    service = RazorpayRiskService(
        adapter=selected_adapter,
        store=RazorpayStore(
            tmp_path / "razorpay.sqlite3"
        ),
        scorer=(
            scorer or default_scorer
        ),
    )

    return service, selected_adapter, calls


def test_order_intent_is_strict() -> None:
    with pytest.raises(ValidationError):
        RazorpayOrderIntent.model_validate(
            {
                "amount": 250000,
                "receipt": "radar_demo_001",
                "customer_email": (
                    "forbidden@example.com"
                ),
            }
        )


def test_order_is_created_once(
    tmp_path,
) -> None:
    service, adapter, _ = _service(
        tmp_path
    )

    intent = RazorpayOrderIntent(
        amount=250000,
        receipt="radar_demo_001",
    )

    first = service.create_order(
        intent,
        idempotency_key="request_0001",
    )

    second = service.create_order(
        intent,
        idempotency_key="request_0001",
    )

    assert first.delivery_mode == "CREATED"

    assert (
        second.delivery_mode
        == "IDEMPOTENT_REPLAY"
    )

    assert first.order_id == second.order_id
    assert adapter.create_calls == 1


def test_public_order_response_contains_no_secret(
    tmp_path,
) -> None:
    service, _, _ = _service(
        tmp_path
    )

    result = service.create_order(
        RazorpayOrderIntent(
            amount=250000,
            receipt="radar_demo_001",
        ),
        idempotency_key="request_0001",
    )

    serialized = result.model_dump()

    assert (
        serialized["checkout_key_id"]
        == "rzp_test_public123"
    )

    assert "key_secret" not in serialized
    assert "notes" not in serialized
    assert "never_expose" not in str(serialized)
    assert result.real_money_used is False
    assert result.action_executed is False


def test_order_key_cannot_change_request(
    tmp_path,
) -> None:
    service, _, _ = _service(
        tmp_path
    )

    service.create_order(
        RazorpayOrderIntent(
            amount=250000,
            receipt="radar_demo_001",
        ),
        idempotency_key="request_0001",
    )

    with pytest.raises(
        RazorpayIdempotencyConflictError
    ):
        service.create_order(
            RazorpayOrderIntent(
                amount=500000,
                receipt="radar_demo_002",
            ),
            idempotency_key="request_0001",
        )


def test_failed_order_is_not_retried_automatically(
    tmp_path,
) -> None:
    adapter = FakeAdapter()
    adapter.fail_create = True

    service, _, _ = _service(
        tmp_path,
        adapter=adapter,
    )

    intent = RazorpayOrderIntent(
        amount=250000,
        receipt="radar_demo_001",
    )

    with pytest.raises(
        RazorpayServiceUnavailableError
    ):
        service.create_order(
            intent,
            idempotency_key="request_0001",
        )

    with pytest.raises(
        RazorpayOperationFailedError
    ):
        service.create_order(
            intent,
            idempotency_key="request_0001",
        )

    assert adapter.create_calls == 1


def test_payment_is_normalized_and_scored(
    tmp_path,
) -> None:
    service, adapter, calls = _service(
        tmp_path
    )

    result = service.score_payment(
        "pay_Test123",
        _context(),
    )

    assert result.delivery_mode == "CALCULATED"
    assert result.provider_status == "captured"
    assert result.provider_method == "card"
    assert result.score.calibrated_probability == 0.02
    assert adapter.fetch_calls == 1
    assert calls["score"] == 1


def test_payment_score_is_idempotently_replayed(
    tmp_path,
) -> None:
    service, _, calls = _service(
        tmp_path
    )

    first = service.score_payment(
        "pay_Test123",
        _context(),
    )

    second = service.score_payment(
        "pay_Test123",
        _context(),
    )

    assert first.delivery_mode == "CALCULATED"

    assert (
        second.delivery_mode
        == "IDEMPOTENT_REPLAY"
    )

    assert calls["score"] == 1

    assert (
        first.score.model_dump()
        == second.score.model_dump()
    )


def test_payment_context_cannot_change_after_score(
    tmp_path,
) -> None:
    service, _, _ = _service(
        tmp_path
    )

    service.score_payment(
        "pay_Test123",
        _context(),
    )

    changed = _context(
        card_network="Mastercard"
    )

    with pytest.raises(
        RazorpayIdempotencyConflictError
    ):
        service.score_payment(
            "pay_Test123",
            changed,
        )


def test_payment_response_contains_no_identity_or_outcome(
    tmp_path,
) -> None:
    service, _, _ = _service(
        tmp_path
    )

    result = service.score_payment(
        "pay_Test123",
        _context(),
    )

    serialized = result.model_dump_json()

    forbidden = {
        "customer_id",
        "customer_email",
        "contact",
        "chargeback_within_120d",
        "true_fraud",
        "dispute_status",
        "dispute_reason",
        "refund_status",
    }

    for value in forbidden:
        assert value not in serialized

    assert "must-not-survive" not in serialized
    assert "+919999999999" not in serialized


def test_test_mode_result_executes_nothing(
    tmp_path,
) -> None:
    service, _, _ = _service(
        tmp_path
    )

    result = service.score_payment(
        "pay_Test123",
        _context(),
    )

    assert result.source == "RAZORPAY_TEST_MODE"

    assert (
        result.synthetic_evaluation_affected
        is False
    )

    assert (
        result.human_approval_required
        is True
    )

    assert (
        result.financial_action_executed
        is False
    )

    assert result.score.action_executed is False