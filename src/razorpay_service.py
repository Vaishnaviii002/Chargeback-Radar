from __future__ import annotations

from typing import Callable, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from src.razorpay_checkout import (
    RazorpayCheckoutInputError,
    RazorpayCheckoutSignatureError,
    RazorpayCheckoutVerificationResult,
    verify_checkout_signature,
)

from src.razorpay_adapter import (
    RazorpayAdapter,
    RazorpayAdapterError,
    RazorpayOrder,
    RazorpayOrderCreateRequest,
)
from src.razorpay_normalizer import (
    NORMALIZATION_VERSION,
    RazorpayMerchantRiskContext,
    RazorpayNormalizationError,
    normalize_razorpay_payment,
)
from src.razorpay_store import (
    RazorpayStore,
    RazorpayStoreCorruptionError,
    RazorpayStoreError,
)
from src.risk_scoring import (
    RiskScoreResult,
    RiskScoringError,
    score_capture_time_payment,
)


class RazorpayServiceError(RuntimeError):
    pass


class RazorpayServiceUnavailableError(
    RazorpayServiceError
):
    pass


class RazorpayServiceValidationError(
    RazorpayServiceError
):
    pass

class RazorpayServiceAuthenticationError(
    RazorpayServiceError
):
    pass


class RazorpayOperationInProgressError(
    RazorpayServiceError
):
    pass


class RazorpayOperationFailedError(
    RazorpayServiceError
):
    pass


class RazorpayOrderIntent(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    amount: int = Field(
        ge=100,
        le=10_000_000,
    )

    currency: Literal["INR"] = "INR"

    receipt: str = Field(
        min_length=1,
        max_length=40,
        pattern=r"^[A-Za-z0-9_-]+$",
    )


class RazorpayOrderResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    delivery_mode: Literal[
        "CREATED",
        "IDEMPOTENT_REPLAY",
    ]

    checkout_key_id: str
    order_id: str
    amount: int
    currency: Literal["INR"]
    receipt: str
    status: str

    source: Literal[
        "RAZORPAY_TEST_MODE"
    ] = "RAZORPAY_TEST_MODE"

    real_money_used: Literal[
        False
    ] = False

    action_executed: Literal[
        False
    ] = False


class RazorpayPaymentRiskResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    delivery_mode: Literal[
        "CALCULATED",
        "IDEMPOTENT_REPLAY",
    ]

    razorpay_payment_id: str
    razorpay_order_id: str

    provider_status: Literal[
        "authorized",
        "captured",
    ]

    provider_method: Literal[
        "card"
    ] = "card"

    normalization_version: Literal[
        "razorpay-normalizer-v1"
    ] = NORMALIZATION_VERSION

    score: RiskScoreResult

    source: Literal[
        "RAZORPAY_TEST_MODE"
    ] = "RAZORPAY_TEST_MODE"

    synthetic_evaluation_affected: Literal[
        False
    ] = False

    human_approval_required: Literal[
        True
    ] = True

    financial_action_executed: Literal[
        False
    ] = False

    disclosure: Literal[
        "Razorpay Test Mode payment scored separately from the synthetic held-out evaluation. No refund, dispute or customer message was executed."
    ] = (
        "Razorpay Test Mode payment scored separately "
        "from the synthetic held-out evaluation. No "
        "refund, dispute or customer message was executed."
    )


class RazorpayVerifiedPaymentRiskResult(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    verification: (
        RazorpayCheckoutVerificationResult
    )

    risk_result: RazorpayPaymentRiskResult

    source: Literal[
        "RAZORPAY_TEST_MODE"
    ] = "RAZORPAY_TEST_MODE"

    signature_required_before_scoring: Literal[
        True
    ] = True

    synthetic_evaluation_affected: Literal[
        False
    ] = False

    financial_action_executed: Literal[
        False
    ] = False

ScoreFunction = Callable[
    [object],
    RiskScoreResult,
]


class RazorpayRiskService:
    def __init__(
        self,
        *,
        adapter: RazorpayAdapter,
        store: RazorpayStore,
        scorer: ScoreFunction = (
            score_capture_time_payment
        ),
    ) -> None:
        self.adapter = adapter
        self.store = store
        self.scorer = scorer

    def _public_test_key(self) -> str:
        key_id = self.adapter.config.key_id

        if (
            not key_id
            or not key_id.startswith(
                "rzp_test_"
            )
        ):
            raise RazorpayServiceUnavailableError(
                "Razorpay Test Mode key is unavailable."
            )

        return key_id

    def create_order(
        self,
        intent: RazorpayOrderIntent,
        *,
        idempotency_key: str,
    ) -> RazorpayOrderResult:
        provider_request = (
            RazorpayOrderCreateRequest(
                amount=intent.amount,
                currency=intent.currency,
                receipt=intent.receipt,
                notes={
                    "source": "chargeback_radar",
                    "environment": "test_mode",
                },
            )
        )

        request_payload = (
            provider_request.model_dump(
                mode="json"
            )
        )

        reservation = self.store.reserve_order(
            idempotency_key,
            request_payload,
        )

        if reservation.state == "PENDING":
            raise RazorpayOperationInProgressError(
                "Order creation is already in progress."
            )

        if reservation.state == "FAILED":
            raise RazorpayOperationFailedError(
                "Previous order creation failed safely; "
                "manual reconciliation is required."
            )

        if reservation.state == "COMPLETED":
            if reservation.response is None:
                raise RazorpayStoreCorruptionError(
                    "Completed order response is missing."
                )

            try:
                order = RazorpayOrder.model_validate(
                    reservation.response
                )
            except ValidationError as error:
                raise RazorpayStoreCorruptionError(
                    "Stored order response is invalid."
                ) from error

            return self._order_result(
                order,
                delivery_mode=(
                    "IDEMPOTENT_REPLAY"
                ),
            )

        try:
            order = self.adapter.create_order(
                provider_request
            )

        except RazorpayAdapterError as error:
            self.store.fail_order(
                idempotency_key,
                request_payload,
                error_code="RAZORPAY_UNAVAILABLE",
            )

            raise RazorpayServiceUnavailableError(
                "Razorpay Test Mode order creation "
                "failed safely."
            ) from error

        self.store.complete_order(
            idempotency_key,
            request_payload,
            order_id=order.id,
            response_payload=order.model_dump(
                mode="json"
            ),
        )

        return self._order_result(
            order,
            delivery_mode="CREATED",
        )

    def _order_result(
        self,
        order: RazorpayOrder,
        *,
        delivery_mode: Literal[
            "CREATED",
            "IDEMPOTENT_REPLAY",
        ],
    ) -> RazorpayOrderResult:
        if order.currency.upper() != "INR":
            raise RazorpayServiceValidationError(
                "Razorpay order currency is not INR."
            )

        if not order.receipt:
            raise RazorpayServiceValidationError(
                "Razorpay order receipt is missing."
            )

        return RazorpayOrderResult(
            delivery_mode=delivery_mode,
            checkout_key_id=(
                self._public_test_key()
            ),
            order_id=order.id,
            amount=order.amount,
            currency="INR",
            receipt=order.receipt,
            status=order.status,
            real_money_used=False,
            action_executed=False,
        )

    def verify_checkout_payment(
        self,
        *,
        server_order_id: str,
        razorpay_payment_id: str,
        razorpay_signature: str,
    ) -> RazorpayCheckoutVerificationResult:
        """
        Verify Checkout authentication first, then bind the
        verified IDs to Razorpay's server-side Test Mode records.

        This method does not capture, refund or otherwise modify
        the payment.
        """

        key_secret = getattr(
            self.adapter.config,
            "key_secret",
            None,
        )

        if key_secret is None:
            raise RazorpayServiceUnavailableError(
                "Razorpay key secret is unavailable."
            )

        try:
            verification = (
                verify_checkout_signature(
                    server_order_id=(
                        server_order_id
                    ),
                    razorpay_payment_id=(
                        razorpay_payment_id
                    ),
                    razorpay_signature=(
                        razorpay_signature
                    ),
                    key_secret=key_secret,
                )
            )

        except RazorpayCheckoutSignatureError as error:
            raise RazorpayServiceAuthenticationError(
                "Razorpay Checkout signature "
                "verification failed."
            ) from error

        except RazorpayCheckoutInputError as error:
            raise RazorpayServiceValidationError(
                str(error)
            ) from error

        try:
            order = self.adapter.fetch_order(
                server_order_id
            )

            payment = self.adapter.fetch_payment(
                razorpay_payment_id
            )

        except RazorpayAdapterError as error:
            raise RazorpayServiceUnavailableError(
                "Verified Razorpay Test Mode records "
                "could not be fetched."
            ) from error

        if order.id != server_order_id:
            raise RazorpayServiceAuthenticationError(
                "Razorpay order identity mismatch."
            )

        if payment.id != razorpay_payment_id:
            raise RazorpayServiceAuthenticationError(
                "Razorpay payment identity mismatch."
            )

        if payment.order_id != server_order_id:
            raise RazorpayServiceAuthenticationError(
                "Razorpay payment does not belong to "
                "the verified order."
            )

        if (
            order.currency.upper() != "INR"
            or payment.currency.upper() != "INR"
        ):
            raise RazorpayServiceValidationError(
                "Only INR Test Mode payments are supported."
            )

        if payment.amount != order.amount:
            raise RazorpayServiceValidationError(
                "Razorpay payment amount does not match "
                "the verified order."
            )

        if payment.method != "card":
            raise RazorpayServiceValidationError(
                "Only card payments can enter the "
                "current risk-scoring workflow."
            )

        if payment.status not in {
            "authorized",
            "captured",
        }:
            raise RazorpayServiceValidationError(
                "Razorpay payment is not authorized "
                "or captured."
            )

        return verification

    def verify_and_score_checkout(
        self,
        *,
        server_order_id: str,
        razorpay_payment_id: str,
        razorpay_signature: str,
        context: RazorpayMerchantRiskContext,
    ) -> RazorpayVerifiedPaymentRiskResult:
        """
        Authenticate the Checkout response before allowing
        the Test Mode payment to enter risk scoring.
        """

        verification = (
            self.verify_checkout_payment(
                server_order_id=server_order_id,
                razorpay_payment_id=(
                    razorpay_payment_id
                ),
                razorpay_signature=(
                    razorpay_signature
                ),
            )
        )

        risk_result = self.score_payment(
            razorpay_payment_id,
            context,
        )

        if (
            risk_result.razorpay_order_id
            != verification.razorpay_order_id
        ):
            raise RazorpayServiceAuthenticationError(
                "Scored payment order does not match "
                "the verified Checkout order."
            )

        if (
            risk_result.razorpay_payment_id
            != verification.razorpay_payment_id
        ):
            raise RazorpayServiceAuthenticationError(
                "Scored payment identity does not match "
                "the verified Checkout payment."
            )

        return RazorpayVerifiedPaymentRiskResult(
            verification=verification,
            risk_result=risk_result,
            source="RAZORPAY_TEST_MODE",
            signature_required_before_scoring=True,
            synthetic_evaluation_affected=False,
            financial_action_executed=False,
    )

    def score_payment(
        self,
        payment_id: str,
        context: RazorpayMerchantRiskContext,
    ) -> RazorpayPaymentRiskResult:
        try:
            payment = self.adapter.fetch_payment(
                payment_id
            )

        except RazorpayAdapterError as error:
            raise RazorpayServiceUnavailableError(
                "Razorpay Test Mode payment could not "
                "be fetched."
            ) from error

        try:
            normalized = (
                normalize_razorpay_payment(
                    payment,
                    context,
                )
            )
        except RazorpayNormalizationError as error:
            raise RazorpayServiceValidationError(
                str(error)
            ) from error

        score_input = (
            normalized.scoring_payload.model_dump(
                mode="json"
            )
        )

        try:
            cached = self.store.load_payment_score(
                payment.id,
                score_input,
            )
        except RazorpayStoreError:
            raise

        if cached is not None:
            try:
                cached_result = (
                    RazorpayPaymentRiskResult.model_validate(
                        cached
                    )
                )
            except ValidationError as error:
                raise RazorpayStoreCorruptionError(
                    "Stored payment score is invalid."
                ) from error

            return cached_result.model_copy(
                update={
                    "delivery_mode": (
                        "IDEMPOTENT_REPLAY"
                    )
                }
            )

        try:
            score = self.scorer(
                normalized.scoring_payload
            )
        except RiskScoringError as error:
            raise RazorpayServiceUnavailableError(
                "Risk scoring is temporarily unavailable."
            ) from error

        result = RazorpayPaymentRiskResult(
            delivery_mode="CALCULATED",
            razorpay_payment_id=(
                normalized.razorpay_payment_id
            ),
            razorpay_order_id=(
                normalized.razorpay_order_id
            ),
            provider_status=(
                normalized.provider_status
            ),
            provider_method="card",
            score=score,
            synthetic_evaluation_affected=False,
            human_approval_required=True,
            financial_action_executed=False,
        )

        result_payload = result.model_dump(
            mode="json"
        )

        stored_payload = (
            self.store.save_payment_score(
                payment.id,
                score_input,
                result_payload,
            )
        )

        if stored_payload != result_payload:
            try:
                stored_result = (
                    RazorpayPaymentRiskResult.model_validate(
                        stored_payload
                    )
                )
            except ValidationError as error:
                raise RazorpayStoreCorruptionError(
                    "Stored payment score is invalid."
                ) from error

            return stored_result.model_copy(
                update={
                    "delivery_mode": (
                        "IDEMPOTENT_REPLAY"
                    )
                }
            )

        return result