from __future__ import annotations

import os
import re
from typing import Any, Literal

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
)


RAZORPAY_API_BASE_URL = "https://api.razorpay.com/v1"

ORDER_ID_PATTERN = re.compile(r"^order_[A-Za-z0-9]+$")
PAYMENT_ID_PATTERN = re.compile(r"^pay_[A-Za-z0-9]+$")


class RazorpayAdapterError(RuntimeError):
    """Base error for the isolated Razorpay integration."""


class RazorpayDisabledError(RazorpayAdapterError):
    pass


class RazorpayConfigurationError(RazorpayAdapterError):
    pass


class RazorpayUnavailableError(RazorpayAdapterError):
    pass


class RazorpayNotFoundError(RazorpayAdapterError):
    pass


class RazorpayResponseError(RazorpayAdapterError):
    pass


def _environment_boolean(
    name: str,
    default: bool,
) -> bool:
    raw = os.getenv(
        name,
        "true" if default else "false",
    )

    return raw.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class RazorpayConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    enabled: bool = False
    test_mode: bool = True
    key_id: str | None = None
    key_secret: SecretStr | None = None

    timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=30.0,
    )

    @classmethod
    def from_env(cls) -> "RazorpayConfig":
        try:
            from dotenv import load_dotenv

            load_dotenv(override=False)
        except ImportError:
            pass

        key_secret = os.getenv(
            "RAZORPAY_KEY_SECRET"
        )

        try:
            return cls(
                enabled=_environment_boolean(
                    "RAZORPAY_INTEGRATION_ENABLED",
                    False,
                ),
                test_mode=_environment_boolean(
                    "RAZORPAY_TEST_MODE",
                    True,
                ),
                key_id=(
                    os.getenv("RAZORPAY_KEY_ID")
                    or None
                ),
                key_secret=(
                    SecretStr(key_secret)
                    if key_secret
                    else None
                ),
                timeout_seconds=float(
                    os.getenv(
                        "RAZORPAY_TIMEOUT_SECONDS",
                        "10",
                    )
                ),
            )
        except (ValueError, ValidationError) as error:
            raise RazorpayConfigurationError(
                "Razorpay configuration is invalid."
            ) from error

    def require_test_credentials(
        self,
    ) -> tuple[str, str]:
        if not self.enabled:
            raise RazorpayDisabledError(
                "Razorpay integration is disabled."
            )

        if not self.test_mode:
            raise RazorpayConfigurationError(
                "Chargeback Radar permits only Razorpay "
                "Test Mode."
            )

        if not self.key_id:
            raise RazorpayConfigurationError(
                "RAZORPAY_KEY_ID is missing."
            )

        if not self.key_id.startswith("rzp_test_"):
            raise RazorpayConfigurationError(
                "RAZORPAY_KEY_ID must be a Test Mode key."
            )

        if self.key_secret is None:
            raise RazorpayConfigurationError(
                "RAZORPAY_KEY_SECRET is missing."
            )

        secret = self.key_secret.get_secret_value()

        if not secret:
            raise RazorpayConfigurationError(
                "RAZORPAY_KEY_SECRET is empty."
            )

        return self.key_id, secret


class RazorpayOrderCreateRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    amount: int = Field(
        ge=100,
        le=10_000_000,
        description="Amount in paise.",
    )

    currency: Literal["INR"] = "INR"

    receipt: str = Field(
        min_length=1,
        max_length=40,
        pattern=r"^[A-Za-z0-9_-]+$",
    )

    notes: dict[str, str] = Field(
        default_factory=dict,
    )

    @field_validator("notes")
    @classmethod
    def validate_notes(
        cls,
        value: dict[str, str],
    ) -> dict[str, str]:
        if len(value) > 15:
            raise ValueError(
                "Razorpay allows at most 15 notes."
            )

        for key, item in value.items():
            if not key or len(key) > 256:
                raise ValueError(
                    "Note keys must contain 1–256 characters."
                )

            if len(item) > 256:
                raise ValueError(
                    "Note values cannot exceed 256 characters."
                )

        return value


class RazorpayOrder(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
    )

    id: str = Field(
        pattern=r"^order_[A-Za-z0-9]+$",
    )

    entity: Literal["order"]
    amount: int = Field(ge=0)
    amount_paid: int = Field(ge=0)
    amount_due: int = Field(ge=0)
    currency: str
    receipt: str | None = None
    status: str
    attempts: int = Field(ge=0)
    notes: dict[str, Any] = Field(
        default_factory=dict,
    )
    created_at: int = Field(ge=0)


class RazorpayPayment(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
    )

    id: str = Field(
        pattern=r"^pay_[A-Za-z0-9]+$",
    )

    entity: Literal["payment"]
    amount: int = Field(ge=0)
    currency: str
    status: str
    order_id: str | None = None
    method: str | None = None
    captured: bool
    description: str | None = None
    bank: str | None = None
    wallet: str | None = None
    created_at: int = Field(ge=0)


class RazorpayAdapter:
    """Minimal Test Mode adapter with no financial actions."""

    def __init__(
        self,
        *,
        config: RazorpayConfig | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.config = (
            config or RazorpayConfig.from_env()
        )

        key_id, key_secret = (
            self.config.require_test_credentials()
        )

        self._auth = httpx.BasicAuth(
            key_id,
            key_secret,
        )

        self._owns_client = client is None

        self._client = client or httpx.Client(
            base_url=RAZORPAY_API_BASE_URL,
            timeout=self.config.timeout_seconds,
            headers={
                "Accept": "application/json",
                "User-Agent": (
                    "Chargeback-Radar-Test-Mode/0.1"
                ),
            },
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "RazorpayAdapter":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = self._client.request(
                method,
                path,
                json=json_body,
                auth=self._auth,
            )
        except httpx.TimeoutException as error:
            raise RazorpayUnavailableError(
                "Razorpay Test Mode request timed out."
            ) from error
        except httpx.RequestError as error:
            raise RazorpayUnavailableError(
                "Razorpay Test Mode is temporarily "
                "unavailable."
            ) from error

        if response.status_code == 404:
            raise RazorpayNotFoundError(
                "Razorpay Test Mode resource was not found."
            )

        if response.status_code >= 400:
            request_id = response.headers.get(
                "x-razorpay-request-id"
            )

            suffix = (
                f" Request ID: {request_id}."
                if request_id
                else ""
            )

            raise RazorpayUnavailableError(
                "Razorpay Test Mode returned HTTP "
                f"{response.status_code}.{suffix}"
            )

        try:
            payload = response.json()
        except ValueError as error:
            raise RazorpayResponseError(
                "Razorpay returned invalid JSON."
            ) from error

        if not isinstance(payload, dict):
            raise RazorpayResponseError(
                "Razorpay returned an invalid object."
            )

        return payload

    @staticmethod
    def _validate_order_id(
        order_id: str,
    ) -> None:
        if not ORDER_ID_PATTERN.fullmatch(order_id):
            raise ValueError(
                "Invalid Razorpay order ID."
            )

    @staticmethod
    def _validate_payment_id(
        payment_id: str,
    ) -> None:
        if not PAYMENT_ID_PATTERN.fullmatch(payment_id):
            raise ValueError(
                "Invalid Razorpay payment ID."
            )

    def create_order(
        self,
        request: RazorpayOrderCreateRequest,
    ) -> RazorpayOrder:
        payload = self._request(
            "POST",
            "/orders",
            json_body=request.model_dump(),
        )

        try:
            return RazorpayOrder.model_validate(
                payload
            )
        except ValidationError as error:
            raise RazorpayResponseError(
                "Razorpay returned an invalid order."
            ) from error

    def fetch_order(
        self,
        order_id: str,
    ) -> RazorpayOrder:
        self._validate_order_id(order_id)

        payload = self._request(
            "GET",
            f"/orders/{order_id}",
        )

        try:
            return RazorpayOrder.model_validate(
                payload
            )
        except ValidationError as error:
            raise RazorpayResponseError(
                "Razorpay returned an invalid order."
            ) from error

    def fetch_payment(
        self,
        payment_id: str,
    ) -> RazorpayPayment:
        self._validate_payment_id(payment_id)

        payload = self._request(
            "GET",
            f"/payments/{payment_id}",
        )

        try:
            return RazorpayPayment.model_validate(
                payload
            )
        except ValidationError as error:
            raise RazorpayResponseError(
                "Razorpay returned an invalid payment."
            ) from error
