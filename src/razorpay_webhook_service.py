from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    SecretStr,
)

from src.razorpay_store import (
    RazorpayStore,
    RazorpayStoreCorruptionError,
    raw_body_sha256,
)
from src.razorpay_webhook import (
    RazorpayVerifiedWebhook,
    RazorpayWebhookConfigurationError,
    parse_verified_webhook,
)


class RazorpayWebhookServiceError(
    RuntimeError
):
    pass


class RazorpayWebhookDisabledError(
    RazorpayWebhookServiceError
):
    pass


@dataclass(frozen=True)
class RazorpayWebhookConfig:
    enabled: bool = True
    webhook_secret: SecretStr | None = None

    @classmethod
    def from_env(
        cls,
    ) -> "RazorpayWebhookConfig":
        try:
            from dotenv import load_dotenv

            load_dotenv(
                override=False
            )

        except ImportError:
            pass

        enabled_value = os.getenv(
            "RAZORPAY_WEBHOOK_ENABLED",
            "true",
        ).strip().lower()

        secret_value = os.getenv(
            "RAZORPAY_WEBHOOK_SECRET"
        )

        return cls(
            enabled=enabled_value in {
                "1",
                "true",
                "yes",
                "on",
            },
            webhook_secret=(
                SecretStr(secret_value)
                if secret_value
                else None
            ),
        )


class RazorpayWebhookDeliveryResult(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    delivery_mode: Literal[
        "PROCESSED",
        "IGNORED",
        "IDEMPOTENT_REPLAY",
    ]

    event: RazorpayVerifiedWebhook

    duplicate: bool

    stored_status: Literal[
        "PROCESSED"
    ] = "PROCESSED"

    human_approval_required: Literal[
        True
    ] = True

    financial_action_executed: Literal[
        False
    ] = False

    disclosure: Literal[
        "Verified Razorpay webhook recorded for defensive monitoring only. No payment, refund, dispute or customer action was executed."
    ] = (
        "Verified Razorpay webhook recorded for "
        "defensive monitoring only. No payment, "
        "refund, dispute or customer action was "
        "executed."
    )


class RazorpayWebhookService:
    def __init__(
        self,
        *,
        store: RazorpayStore,
        config: (
            RazorpayWebhookConfig | None
        ) = None,
    ) -> None:
        self.store = store

        self.config = (
            config
            or RazorpayWebhookConfig.from_env()
        )

    def process(
        self,
        *,
        raw_body: bytes,
        signature: str,
        event_id: str,
    ) -> RazorpayWebhookDeliveryResult:
        if not self.config.enabled:
            raise RazorpayWebhookDisabledError(
                "Razorpay webhook processing is "
                "disabled."
            )

        webhook_secret = (
            self.config.webhook_secret
        )

        if webhook_secret is None:
            raise (
                RazorpayWebhookConfigurationError(
                    "Razorpay webhook secret is "
                    "unavailable."
                )
            )

        verified_event = (
            parse_verified_webhook(
                raw_body=raw_body,
                signature=signature,
                event_id=event_id,
                webhook_secret=webhook_secret,
            )
        )

        payload_hash = raw_body_sha256(
            raw_body
        )

        is_new_event = (
            self.store.reserve_webhook_event(
                event_id=(
                    verified_event.event_id
                ),
                payload_hash=payload_hash,
                event_type=(
                    verified_event.event_type
                ),
            )
        )

        if not is_new_event:
            existing_status = (
                self.store.webhook_event_status(
                    verified_event.event_id
                )
            )

            if existing_status not in {
                "RECEIVED",
                "PROCESSED",
            }:
                raise RazorpayStoreCorruptionError(
                    "Stored webhook state is invalid."
                )

            if existing_status == "RECEIVED":
                # Processing is currently only a safe,
                # deterministic classification with no
                # external action, so recovery is safe.
                self.store.complete_webhook_event(
                    verified_event.event_id
                )

            return RazorpayWebhookDeliveryResult(
                delivery_mode=(
                    "IDEMPOTENT_REPLAY"
                ),
                event=verified_event,
                duplicate=True,
                stored_status="PROCESSED",
                human_approval_required=True,
                financial_action_executed=False,
            )

        self.store.complete_webhook_event(
            verified_event.event_id
        )

        delivery_mode: Literal[
            "PROCESSED",
            "IGNORED",
            "IDEMPOTENT_REPLAY",
        ] = (
            "PROCESSED"
            if (
                verified_event
                .processing_decision
                == "PROCESS"
            )
            else "IGNORED"
        )

        return RazorpayWebhookDeliveryResult(
            delivery_mode=delivery_mode,
            event=verified_event,
            duplicate=False,
            stored_status="PROCESSED",
            human_approval_required=True,
            financial_action_executed=False,
        )