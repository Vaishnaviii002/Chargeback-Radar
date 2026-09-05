from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets

import httpx
from dotenv import load_dotenv


BASE_URL = "http://127.0.0.1:8000"


def enabled(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def main() -> None:
    load_dotenv(override=False)

    if not enabled("RAZORPAY_TEST_MODE"):
        raise SystemExit(
            "Refusing to run outside Razorpay Test Mode."
        )

    if not enabled("RAZORPAY_WEBHOOK_ENABLED"):
        raise SystemExit(
            "Set RAZORPAY_WEBHOOK_ENABLED=true in local .env."
        )

    secret = os.getenv("RAZORPAY_WEBHOOK_SECRET")

    if not secret:
        raise SystemExit(
            "Configure RAZORPAY_WEBHOOK_SECRET in local .env."
        )

    event_id = "event_demo_" + secrets.token_hex(8)
    document = {
        "entity": "event",
        "event": "payment.captured",
        "created_at": 1_788_550_000,
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_DemoPayment12345",
                    "order_id": "order_DemoOrder12345",
                }
            }
        },
    }
    raw_body = json.dumps(
        document,
        separators=(",", ":"),
    ).encode("utf-8")
    signature = hmac.new(
        secret.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()
    headers = {
        "Content-Type": "application/json",
        "X-Razorpay-Signature": signature,
        "X-Razorpay-Event-Id": event_id,
    }

    with httpx.Client(
        base_url=BASE_URL,
        timeout=10,
    ) as client:
        first = client.post(
            "/api/razorpay-test/webhooks",
            content=raw_body,
            headers=headers,
        )
        second = client.post(
            "/api/razorpay-test/webhooks",
            content=raw_body,
            headers=headers,
        )

    first.raise_for_status()
    second.raise_for_status()
    first_result = first.json()
    second_result = second.json()

    print(
        "First delivery:",
        first_result["delivery_mode"],
    )
    print(
        "Repeated delivery:",
        second_result["delivery_mode"],
    )
    print(
        "Human approval required:",
        first_result["human_approval_required"],
    )
    print(
        "Financial action executed:",
        first_result["financial_action_executed"],
    )


if __name__ == "__main__":
    main()
