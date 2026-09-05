from __future__ import annotations

from functools import lru_cache
import os

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Request,
    status,
)
from pydantic import (
    BaseModel,
    ConfigDict,
)

from src.razorpay_checkout import (
    RazorpayCheckoutVerificationRequest,
    RazorpayCheckoutVerificationResult,
)

from src.razorpay_adapter import (
    RazorpayAdapter,
    RazorpayAdapterError,
    RazorpayConfig,
    RazorpayNotFoundError,
)
from src.razorpay_normalizer import (
    RazorpayMerchantRiskContext,
)
from src.razorpay_service import (
    RazorpayOperationFailedError,
    RazorpayOperationInProgressError,
    RazorpayOrderIntent,
    RazorpayOrderResult,
    RazorpayRiskService,
    RazorpayServiceAuthenticationError,
    RazorpayServiceUnavailableError,
    RazorpayServiceValidationError,
    RazorpayVerifiedPaymentRiskResult,
)
from src.razorpay_store import (
    RazorpayIdempotencyConflictError,
    RazorpayStore,
    RazorpayStoreError,
)
from src.razorpay_webhook import (
    MAX_WEBHOOK_BODY_BYTES,
    RazorpayWebhookConfigurationError,
    RazorpayWebhookPayloadError,
    RazorpayWebhookSignatureError,
)
from src.razorpay_webhook_service import (
    RazorpayWebhookDeliveryResult,
    RazorpayWebhookDisabledError,
    RazorpayWebhookService,
)


class RazorpaySystemStatus(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    status: str
    enabled: bool
    test_mode: bool
    credentials_configured: bool
    webhook_secret_configured: bool
    automatic_financial_actions: bool
    synthetic_evaluation_separate: bool


class RazorpayOrderLookupResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    order_id: str
    amount: int
    amount_paid: int
    amount_due: int
    currency: str
    receipt: str | None
    status: str
    attempts: int

    source: str = "RAZORPAY_TEST_MODE"
    real_money_used: bool = False
    action_executed: bool = False


class RazorpayPaymentLookupResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    payment_id: str
    order_id: str | None
    amount: int
    currency: str
    status: str
    method: str | None
    captured: bool
    created_at: int

    source: str = "RAZORPAY_TEST_MODE"
    synthetic_evaluation_separate: bool = True
    action_executed: bool = False


class RazorpayVerifyAndScoreRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    checkout: RazorpayCheckoutVerificationRequest
    context: RazorpayMerchantRiskContext

@lru_cache(maxsize=1)
def get_razorpay_config() -> RazorpayConfig:
    try:
        return RazorpayConfig.from_env()
    except RazorpayAdapterError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Razorpay Test Mode configuration is "
                "unavailable."
            ),
        ) from error


@lru_cache(maxsize=1)
def get_razorpay_adapter() -> RazorpayAdapter:
    try:
        return RazorpayAdapter(
            config=get_razorpay_config()
        )
    except RazorpayAdapterError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Razorpay Test Mode integration is "
                "unavailable."
            ),
        ) from error


@lru_cache(maxsize=1)
def get_razorpay_store() -> RazorpayStore:
    return RazorpayStore()


@lru_cache(maxsize=1)
def get_razorpay_service() -> RazorpayRiskService:
    return RazorpayRiskService(
        adapter=get_razorpay_adapter(),
        store=get_razorpay_store(),
    )

@lru_cache(maxsize=1)
def get_razorpay_webhook_service(
) -> RazorpayWebhookService:
    return RazorpayWebhookService(
        store=get_razorpay_store(),
    )

router = APIRouter(
    prefix="/api/razorpay-test",
    tags=["Razorpay Test Mode"],
)


async def _read_bounded_raw_body(
    request: Request,
) -> bytes:
    raw_body = bytearray()

    async for chunk in request.stream():
        if (
            len(raw_body) + len(chunk)
            > MAX_WEBHOOK_BODY_BYTES
        ):
            raise RazorpayWebhookPayloadError(
                "Webhook body exceeds the size limit."
            )

        raw_body.extend(chunk)

    return bytes(raw_body)


@router.get(
    "/status",
    response_model=RazorpaySystemStatus,
)
def razorpay_status(
    config: RazorpayConfig = Depends(
        get_razorpay_config
    ),
) -> RazorpaySystemStatus:
    credentials_configured = bool(
        config.key_id
        and config.key_id.startswith(
            "rzp_test_"
        )
        and config.key_secret is not None
    )

    if not config.enabled:
        system_status = "disabled"
    elif (
        not config.test_mode
        or not credentials_configured
    ):
        system_status = "misconfigured"
    else:
        system_status = "ready"

    return RazorpaySystemStatus(
        status=system_status,
        enabled=config.enabled,
        test_mode=config.test_mode,
        credentials_configured=(
            credentials_configured
        ),
        webhook_secret_configured=bool(
            os.getenv(
                "RAZORPAY_WEBHOOK_SECRET"
            )
        ),
        automatic_financial_actions=False,
        synthetic_evaluation_separate=True,
    )


@router.post(
    "/webhooks",
    response_model=RazorpayWebhookDeliveryResult,
    responses={
        400: {
            "description": (
                "Malformed verified webhook payload"
            )
        },
        401: {
            "description": (
                "Webhook signature verification failed"
            )
        },
        409: {
            "description": (
                "Webhook event ID conflict"
            )
        },
        503: {
            "description": (
                "Webhook processing unavailable"
            )
        },
    },
)
async def receive_razorpay_webhook(
    request: Request,
    razorpay_signature: str | None = Header(
        default=None,
        alias="X-Razorpay-Signature",
    ),
    razorpay_event_id: str = Header(
        default="",
        alias="X-Razorpay-Event-Id",
    ),
    service: RazorpayWebhookService = Depends(
        get_razorpay_webhook_service
    ),
) -> RazorpayWebhookDeliveryResult:
    try:
        if razorpay_signature is None:
            raise RazorpayWebhookSignatureError(
                "Webhook signature is missing."
            )

        # Stream into a bounded buffer so the exact bytes reach
        # signature verification without accepting an unbounded
        # request body. Do not declare a Pydantic body model.
        raw_body = await _read_bounded_raw_body(
            request
        )

        return service.process(
            raw_body=raw_body,
            signature=razorpay_signature,
            event_id=razorpay_event_id,
        )

    except RazorpayWebhookSignatureError as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Razorpay webhook signature "
                "verification failed."
            ),
        ) from error

    except RazorpayWebhookPayloadError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Razorpay webhook payload is invalid."
            ),
        ) from error

    except RazorpayIdempotencyConflictError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Razorpay webhook event ID conflict."
            ),
        ) from error

    except (
        RazorpayWebhookConfigurationError,
        RazorpayWebhookDisabledError,
        RazorpayStoreError,
    ) as error:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Razorpay webhook processing is "
                "temporarily unavailable."
            ),
        ) from error

@router.post(
    "/orders",
    response_model=RazorpayOrderResult,
    responses={
        409: {
            "description": (
                "Idempotency conflict or operation "
                "already in progress"
            )
        },
        503: {
            "description": (
                "Razorpay Test Mode unavailable"
            )
        },
    },
)
def create_razorpay_order(
    intent: RazorpayOrderIntent,
    idempotency_key: str = Header(
        ...,
        alias="Idempotency-Key",
        min_length=8,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
    ),
    service: RazorpayRiskService = Depends(
        get_razorpay_service
    ),
) -> RazorpayOrderResult:
    try:
        return service.create_order(
            intent,
            idempotency_key=idempotency_key,
        )

    except (
        RazorpayIdempotencyConflictError,
        RazorpayOperationInProgressError,
    ) as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Order idempotency conflict or "
                "operation already in progress."
            ),
        ) from error

    except (
        RazorpayOperationFailedError,
        RazorpayServiceUnavailableError,
        RazorpayStoreError,
    ) as error:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Razorpay Test Mode order creation "
                "failed safely."
            ),
        ) from error


@router.get(
    "/orders/{order_id}",
    response_model=RazorpayOrderLookupResult,
)
def fetch_razorpay_order(
    order_id: str,
    adapter: RazorpayAdapter = Depends(
        get_razorpay_adapter
    ),
) -> RazorpayOrderLookupResult:
    try:
        order = adapter.fetch_order(
            order_id
        )

    except ValueError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_CONTENT
            ),
            detail="Invalid Razorpay order ID.",
        ) from error

    except RazorpayNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Razorpay Test order not found.",
        ) from error

    except RazorpayAdapterError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Razorpay Test Mode is temporarily "
                "unavailable."
            ),
        ) from error

    return RazorpayOrderLookupResult(
        order_id=order.id,
        amount=order.amount,
        amount_paid=order.amount_paid,
        amount_due=order.amount_due,
        currency=order.currency,
        receipt=order.receipt,
        status=order.status,
        attempts=order.attempts,
        source="RAZORPAY_TEST_MODE",
        real_money_used=False,
        action_executed=False,
    )

@router.post(
    "/orders/{order_id}/verify-checkout",
    response_model=(
        RazorpayCheckoutVerificationResult
    ),
    responses={
        401: {
            "description": (
                "Checkout signature verification failed"
            )
        },
        422: {
            "description": (
                "Invalid Checkout payment data"
            )
        },
        503: {
            "description": (
                "Razorpay Test Mode unavailable"
            )
        },
    },
)
def verify_razorpay_checkout(
    order_id: str,
    request: RazorpayCheckoutVerificationRequest,
    service: RazorpayRiskService = Depends(
        get_razorpay_service
    ),
) -> RazorpayCheckoutVerificationResult:
    try:
        return service.verify_checkout_payment(
            server_order_id=order_id,
            razorpay_payment_id=(
                request.razorpay_payment_id
            ),
            razorpay_signature=(
                request.razorpay_signature
            ),
        )

    except RazorpayServiceAuthenticationError as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Razorpay Checkout signature "
                "verification failed."
            ),
        ) from error

    except RazorpayServiceValidationError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_CONTENT
            ),
            detail=str(error),
        ) from error

    except RazorpayServiceUnavailableError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Razorpay Test Mode verification is "
                "temporarily unavailable."
            ),
        ) from error

@router.post(
    "/orders/{order_id}/verify-and-score",
    response_model=(
        RazorpayVerifiedPaymentRiskResult
    ),
    responses={
        401: {
            "description": (
                "Checkout authentication failed"
            )
        },
        409: {
            "description": (
                "Conflicting payment score input"
            )
        },
        422: {
            "description": (
                "Invalid capture-time context"
            )
        },
        503: {
            "description": (
                "Test Mode provider or scoring unavailable"
            )
        },
    },
)
def verify_and_score_razorpay_checkout(
    order_id: str,
    request: RazorpayVerifyAndScoreRequest,
    service: RazorpayRiskService = Depends(
        get_razorpay_service
    ),
) -> RazorpayVerifiedPaymentRiskResult:
    try:
        return service.verify_and_score_checkout(
            server_order_id=order_id,
            razorpay_payment_id=(
                request.checkout.razorpay_payment_id
            ),
            razorpay_signature=(
                request.checkout.razorpay_signature
            ),
            context=request.context,
        )

    except RazorpayServiceAuthenticationError as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Razorpay Checkout authentication "
                "failed."
            ),
        ) from error

    except RazorpayIdempotencyConflictError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Payment was already scored using "
                "different capture-time input."
            ),
        ) from error

    except (
        RazorpayServiceValidationError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_CONTENT
            ),
            detail=(
                "The verified Test Mode payment could "
                "not be scored with this context."
            ),
        ) from error

    except (
        RazorpayServiceUnavailableError,
        RazorpayStoreError,
    ) as error:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Verified Test Mode risk scoring is "
                "temporarily unavailable."
            ),
        ) from error   

@router.get(
    "/payments/{payment_id}",
    response_model=RazorpayPaymentLookupResult,
)
def fetch_razorpay_payment(
    payment_id: str,
    adapter: RazorpayAdapter = Depends(
        get_razorpay_adapter
    ),
) -> RazorpayPaymentLookupResult:
    try:
        payment = adapter.fetch_payment(
            payment_id
        )

    except ValueError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_CONTENT
            ),
            detail="Invalid Razorpay payment ID.",
        ) from error

    except RazorpayNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Razorpay Test payment not found.",
        ) from error

    except RazorpayAdapterError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Razorpay Test Mode is temporarily "
                "unavailable."
            ),
        ) from error

    return RazorpayPaymentLookupResult(
        payment_id=payment.id,
        order_id=payment.order_id,
        amount=payment.amount,
        currency=payment.currency,
        status=payment.status,
        method=payment.method,
        captured=payment.captured,
        created_at=payment.created_at,
        source="RAZORPAY_TEST_MODE",
        synthetic_evaluation_separate=True,
        action_executed=False,
    )
