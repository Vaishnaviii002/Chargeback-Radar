from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
)


UTC = timezone.utc

DEFAULT_DATABASE_PATH = Path(
    "runtime/razorpay.sqlite3"
)

IDEMPOTENCY_KEY_PATTERN = re.compile(
    r"^[A-Za-z0-9_-]{8,64}$"
)

PAYMENT_ID_PATTERN = re.compile(
    r"^pay_[A-Za-z0-9]+$"
)

EVENT_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9_-]{6,128}$"
)

SHA256_PATTERN = re.compile(
    r"^[0-9a-f]{64}$"
)


class RazorpayStoreError(RuntimeError):
    pass


class RazorpayIdempotencyConflictError(
    RazorpayStoreError
):
    pass


class RazorpayStoreCorruptionError(
    RazorpayStoreError
):
    pass


class OrderReservation(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    state: Literal[
        "RESERVED",
        "PENDING",
        "COMPLETED",
        "FAILED",
    ]

    order_id: str | None = None
    response: dict[str, Any] | None = None
    error_code: str | None = None


def _utc_now() -> str:
    return datetime.now(
        UTC
    ).isoformat()


def canonical_json(
    value: dict[str, Any],
) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise RazorpayStoreError(
            "Idempotency payload is not valid JSON."
        ) from error


def payload_sha256(
    value: dict[str, Any],
) -> str:
    return hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()


def raw_body_sha256(
    raw_body: bytes,
) -> str:
    return hashlib.sha256(
        raw_body
    ).hexdigest()


class RazorpayStore:
    """SQLite-backed idempotency store; contains no credentials."""

    def __init__(
        self,
        database_path: str | Path | None = None,
    ) -> None:
        configured_path = (
            os.getenv("RAZORPAY_DATABASE_PATH", "").strip()
            if database_path is None
            else database_path
        )

        self.database_path = Path(
            configured_path or DEFAULT_DATABASE_PATH
        )

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=10,
        )

        connection.row_factory = sqlite3.Row

        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        connection.execute(
            "PRAGMA busy_timeout = 10000"
        )

        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS order_requests (
                    idempotency_key TEXT PRIMARY KEY,
                    request_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    order_id TEXT,
                    response_json TEXT,
                    error_code TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    CHECK (
                        status IN (
                            'PENDING',
                            'COMPLETED',
                            'FAILED'
                        )
                    )
                );

                CREATE TABLE IF NOT EXISTS payment_scores (
                    payment_id TEXT PRIMARY KEY,
                    input_hash TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS webhook_events (
                    event_id TEXT PRIMARY KEY,
                    payload_hash TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    processed_at TEXT,
                    CHECK (
                        status IN (
                            'RECEIVED',
                            'PROCESSED'
                        )
                    )
                );
                """
            )

    @staticmethod
    def _validate_idempotency_key(
        value: str,
    ) -> None:
        if not IDEMPOTENCY_KEY_PATTERN.fullmatch(
            value
        ):
            raise ValueError(
                "Invalid idempotency key."
            )

    @staticmethod
    def _validate_payment_id(
        value: str,
    ) -> None:
        if not PAYMENT_ID_PATTERN.fullmatch(
            value
        ):
            raise ValueError(
                "Invalid Razorpay payment ID."
            )

    @staticmethod
    def _validate_event(
        event_id: str,
        payload_hash: str,
        event_type: str,
    ) -> None:
        if not EVENT_ID_PATTERN.fullmatch(
            event_id
        ):
            raise ValueError(
                "Invalid Razorpay event ID."
            )

        if not SHA256_PATTERN.fullmatch(
            payload_hash
        ):
            raise ValueError(
                "Invalid webhook payload hash."
            )

        if (
            not event_type
            or len(event_type) > 100
        ):
            raise ValueError(
                "Invalid webhook event type."
            )

    @staticmethod
    def _decode_object(
        raw: str | None,
    ) -> dict[str, Any] | None:
        if raw is None:
            return None

        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise RazorpayStoreCorruptionError(
                "Stored JSON is invalid."
            ) from error

        if not isinstance(value, dict):
            raise RazorpayStoreCorruptionError(
                "Stored JSON is not an object."
            )

        return value

    def reserve_order(
        self,
        idempotency_key: str,
        request_payload: dict[str, Any],
    ) -> OrderReservation:
        self._validate_idempotency_key(
            idempotency_key
        )

        request_hash = payload_sha256(
            request_payload
        )

        now = _utc_now()

        with self._connect() as connection:
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            row = connection.execute(
                """
                SELECT
                    request_hash,
                    status,
                    order_id,
                    response_json,
                    error_code
                FROM order_requests
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()

            if row is None:
                connection.execute(
                    """
                    INSERT INTO order_requests (
                        idempotency_key,
                        request_hash,
                        status,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, 'PENDING', ?, ?)
                    """,
                    (
                        idempotency_key,
                        request_hash,
                        now,
                        now,
                    ),
                )

                return OrderReservation(
                    state="RESERVED"
                )

            if (
                row["request_hash"]
                != request_hash
            ):
                raise (
                    RazorpayIdempotencyConflictError(
                        "Idempotency key was reused "
                        "with a different request."
                    )
                )

            status = str(row["status"])

            if status == "COMPLETED":
                return OrderReservation(
                    state="COMPLETED",
                    order_id=row["order_id"],
                    response=self._decode_object(
                        row["response_json"]
                    ),
                )

            if status == "FAILED":
                return OrderReservation(
                    state="FAILED",
                    error_code=row["error_code"],
                )

            return OrderReservation(
                state="PENDING"
            )

    def complete_order(
        self,
        idempotency_key: str,
        request_payload: dict[str, Any],
        *,
        order_id: str,
        response_payload: dict[str, Any],
    ) -> None:
        self._validate_idempotency_key(
            idempotency_key
        )

        request_hash = payload_sha256(
            request_payload
        )

        response_json = canonical_json(
            response_payload
        )

        with self._connect() as connection:
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            row = connection.execute(
                """
                SELECT
                    request_hash,
                    status,
                    order_id
                FROM order_requests
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()

            if row is None:
                raise RazorpayStoreError(
                    "Order reservation does not exist."
                )

            if (
                row["request_hash"]
                != request_hash
            ):
                raise (
                    RazorpayIdempotencyConflictError(
                        "Order request hash changed."
                    )
                )

            if row["status"] == "COMPLETED":
                if row["order_id"] != order_id:
                    raise (
                        RazorpayIdempotencyConflictError(
                            "Completed reservation contains "
                            "a different order."
                        )
                    )

                return

            if row["status"] != "PENDING":
                raise RazorpayStoreError(
                    "Failed order reservation cannot "
                    "be completed."
                )

            connection.execute(
                """
                UPDATE order_requests
                SET
                    status = 'COMPLETED',
                    order_id = ?,
                    response_json = ?,
                    error_code = NULL,
                    updated_at = ?
                WHERE idempotency_key = ?
                """,
                (
                    order_id,
                    response_json,
                    _utc_now(),
                    idempotency_key,
                ),
            )

    def fail_order(
        self,
        idempotency_key: str,
        request_payload: dict[str, Any],
        *,
        error_code: str,
    ) -> None:
        self._validate_idempotency_key(
            idempotency_key
        )

        if (
            not error_code
            or len(error_code) > 100
        ):
            raise ValueError(
                "Invalid safe error code."
            )

        request_hash = payload_sha256(
            request_payload
        )

        with self._connect() as connection:
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            row = connection.execute(
                """
                SELECT request_hash, status
                FROM order_requests
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()

            if row is None:
                raise RazorpayStoreError(
                    "Order reservation does not exist."
                )

            if (
                row["request_hash"]
                != request_hash
            ):
                raise (
                    RazorpayIdempotencyConflictError(
                        "Order request hash changed."
                    )
                )

            if row["status"] == "COMPLETED":
                raise RazorpayStoreError(
                    "Completed order cannot be failed."
                )

            connection.execute(
                """
                UPDATE order_requests
                SET
                    status = 'FAILED',
                    error_code = ?,
                    updated_at = ?
                WHERE idempotency_key = ?
                """,
                (
                    error_code,
                    _utc_now(),
                    idempotency_key,
                ),
            )

    def load_payment_score(
        self,
        payment_id: str,
        input_payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        self._validate_payment_id(
            payment_id
        )

        input_hash = payload_sha256(
            input_payload
        )

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT input_hash, response_json
                FROM payment_scores
                WHERE payment_id = ?
                """,
                (payment_id,),
            ).fetchone()

        if row is None:
            return None

        if row["input_hash"] != input_hash:
            raise RazorpayIdempotencyConflictError(
                "Payment was previously scored with "
                "different capture-time input."
            )

        value = self._decode_object(
            row["response_json"]
        )

        if value is None:
            raise RazorpayStoreCorruptionError(
                "Stored payment score is missing."
            )

        return value

    def save_payment_score(
        self,
        payment_id: str,
        input_payload: dict[str, Any],
        response_payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._validate_payment_id(
            payment_id
        )

        input_hash = payload_sha256(
            input_payload
        )

        response_json = canonical_json(
            response_payload
        )

        with self._connect() as connection:
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            row = connection.execute(
                """
                SELECT input_hash, response_json
                FROM payment_scores
                WHERE payment_id = ?
                """,
                (payment_id,),
            ).fetchone()

            if row is not None:
                if row["input_hash"] != input_hash:
                    raise (
                        RazorpayIdempotencyConflictError(
                            "Payment input changed after "
                            "its first score."
                        )
                    )

                existing = self._decode_object(
                    row["response_json"]
                )

                if existing is None:
                    raise RazorpayStoreCorruptionError(
                        "Stored payment score is missing."
                    )

                return existing

            connection.execute(
                """
                INSERT INTO payment_scores (
                    payment_id,
                    input_hash,
                    response_json,
                    created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    payment_id,
                    input_hash,
                    response_json,
                    _utc_now(),
                ),
            )

        return response_payload

    def reserve_webhook_event(
        self,
        *,
        event_id: str,
        payload_hash: str,
        event_type: str,
    ) -> bool:
        self._validate_event(
            event_id,
            payload_hash,
            event_type,
        )

        with self._connect() as connection:
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            row = connection.execute(
                """
                SELECT payload_hash, event_type
                FROM webhook_events
                WHERE event_id = ?
                """,
                (event_id,),
            ).fetchone()

            if row is not None:
                if (
                    row["payload_hash"]
                    != payload_hash
                    or row["event_type"]
                    != event_type
                ):
                    raise (
                        RazorpayIdempotencyConflictError(
                            "Webhook event ID was reused "
                            "with different content."
                        )
                    )

                return False

            connection.execute(
                """
                INSERT INTO webhook_events (
                    event_id,
                    payload_hash,
                    event_type,
                    status,
                    received_at
                )
                VALUES (?, ?, ?, 'RECEIVED', ?)
                """,
                (
                    event_id,
                    payload_hash,
                    event_type,
                    _utc_now(),
                ),
            )

        return True

    def complete_webhook_event(
        self,
        event_id: str,
    ) -> None:
        if not EVENT_ID_PATTERN.fullmatch(
            event_id
        ):
            raise ValueError(
                "Invalid Razorpay event ID."
            )

        with self._connect() as connection:
            result = connection.execute(
                """
                UPDATE webhook_events
                SET
                    status = 'PROCESSED',
                    processed_at = ?
                WHERE event_id = ?
                """,
                (
                    _utc_now(),
                    event_id,
                ),
            )

            if result.rowcount != 1:
                raise RazorpayStoreError(
                    "Webhook event does not exist."
                )

    def webhook_event_status(
        self,
        event_id: str,
    ) -> str | None:
        if not EVENT_ID_PATTERN.fullmatch(
            event_id
        ):
            raise ValueError(
                "Invalid Razorpay event ID."
            )

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT status
                FROM webhook_events
                WHERE event_id = ?
                """,
                (event_id,),
            ).fetchone()

        return (
            str(row["status"])
            if row is not None
            else None
        )
