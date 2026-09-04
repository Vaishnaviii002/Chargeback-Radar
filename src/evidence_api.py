from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Mapping
from uuid import uuid4

import pandas as pd
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Response,
    status,
)
from pydantic import BaseModel, ConfigDict

from src.evidence_facts import EvidenceFactError, build_evidence_case
from src.evidence_guardrails import EvidenceGuardrailError
from src.evidence_service import EvidenceDeliveryResult, EvidenceService
from src.evidence_store import (
    AuditIntegrityError,
    AuditVerification,
    EvidenceStoreError,
)


UTC = timezone.utc
PAYMENT_ID_PATTERN = re.compile(r"^pay_[A-Za-z0-9_-]{3,96}$")


class EvidenceRepositoryError(RuntimeError):
    pass


class EvidencePaymentNotFoundError(EvidenceRepositoryError):
    pass


class EvidenceGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    as_of: datetime | None = None
    refresh: bool = False


class EvidenceSystemStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    ai_enabled: bool
    api_key_configured: bool
    model: str
    deterministic_fallback_available: bool
    cache_enabled: bool
    audit_integrity: str
    automatic_action_execution: bool


class ParquetEvidenceRepository:
    """Read the held-out payment and operational timeline by payment ID."""

    def __init__(
        self,
        payment_path: str | Path = "reports/test_policy.parquet",
        operation_path: str | Path = "data/operations.parquet",
    ) -> None:
        self.payment_path = Path(payment_path)
        self.operation_path = Path(operation_path)
        self._payments: pd.DataFrame | None = None
        self._operations: pd.DataFrame | None = None

    def _load_payments(self) -> pd.DataFrame:
        if self._payments is None:
            if not self.payment_path.exists():
                raise EvidenceRepositoryError(
                    "The scored payment dataset is unavailable."
                )
            self._payments = pd.read_parquet(self.payment_path)
        return self._payments

    def _load_operations(self) -> pd.DataFrame | None:
        if self._operations is None:
            if not self.operation_path.exists():
                return None
            self._operations = pd.read_parquet(self.operation_path)
        return self._operations

    @staticmethod
    def _one_record(
        frame: pd.DataFrame,
        *,
        payment_id: str,
        record_name: str,
    ) -> dict[str, Any] | None:
        matches = frame.loc[frame["payment_id"] == payment_id]
        if matches.empty:
            return None
        if len(matches) != 1:
            raise EvidenceRepositoryError(
                f"Multiple {record_name} records were found."
            )
        return dict(matches.iloc[0].to_dict())

    def get_case_records(
        self,
        payment_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        payments = self._load_payments()
        payment = self._one_record(
            payments,
            payment_id=payment_id,
            record_name="payment",
        )
        if payment is None:
            raise EvidencePaymentNotFoundError(
                "Payment not found."
            )

        operations = self._load_operations()
        operation = (
            self._one_record(
                operations,
                payment_id=payment_id,
                record_name="operation",
            )
            if operations is not None
            else None
        )
        return payment, operation


@lru_cache(maxsize=1)
def get_evidence_repository() -> ParquetEvidenceRepository:
    return ParquetEvidenceRepository()


@lru_cache(maxsize=1)
def get_evidence_service() -> EvidenceService:
    return EvidenceService()


def _safe_as_of(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC).replace(second=0, microsecond=0)
    if value.tzinfo is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="as_of must include a timezone.",
        )
    normalized = value.astimezone(UTC)
    if normalized > datetime.now(UTC):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="as_of cannot be in the future.",
        )
    return normalized


def _validate_payment_id(payment_id: str) -> None:
    if not PAYMENT_ID_PATTERN.fullmatch(payment_id):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid payment ID format.",
        )


router = APIRouter(
    prefix="/api/evidence",
    tags=["Evidence Copilot"],
)


@router.get(
    "/status",
    response_model=EvidenceSystemStatus,
)
def evidence_status(
    service: EvidenceService = Depends(get_evidence_service),
) -> EvidenceSystemStatus:
    try:
        verification = service.audit.verify()
    except AuditIntegrityError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Evidence audit integrity verification failed.",
        ) from error

    generator_config = getattr(service.generator, "config", None)
    ai_enabled = bool(getattr(generator_config, "enabled", False))
    api_key_configured = bool(
        getattr(generator_config, "api_key", None)
    )
    model = str(getattr(generator_config, "model", "unconfigured"))

    return EvidenceSystemStatus(
        status="ready",
        ai_enabled=ai_enabled,
        api_key_configured=api_key_configured,
        model=model,
        deterministic_fallback_available=True,
        cache_enabled=True,
        audit_integrity=verification.integrity_mode,
        automatic_action_execution=False,
    )


@router.get(
    "/audit/verify",
    response_model=AuditVerification,
)
def verify_evidence_audit(
    service: EvidenceService = Depends(get_evidence_service),
) -> AuditVerification:
    try:
        return service.audit.verify()
    except AuditIntegrityError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Evidence audit integrity verification failed.",
        ) from error


@router.post(
    "/{payment_id}/generate",
    response_model=EvidenceDeliveryResult,
    responses={
        404: {"description": "Payment not found"},
        422: {"description": "Invalid or unsafe evidence request"},
        503: {"description": "Evidence subsystem unavailable"},
    },
)
def generate_transaction_evidence(
    payment_id: str,
    request: EvidenceGenerateRequest,
    response: Response,
    repository: ParquetEvidenceRepository = Depends(
        get_evidence_repository
    ),
    service: EvidenceService = Depends(get_evidence_service),
) -> EvidenceDeliveryResult:
    _validate_payment_id(payment_id)
    response.headers["X-Chargeback-Radar-Request-Id"] = uuid4().hex

    try:
        payment, operation = repository.get_case_records(payment_id)
    except EvidencePaymentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Payment not found.",
        ) from error
    except EvidenceRepositoryError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Evidence data is temporarily unavailable.",
        ) from error

    try:
        evidence_case = build_evidence_case(
            payment,
            operation=operation,
            as_of=_safe_as_of(request.as_of),
        )
        return service.generate(
            evidence_case,
            refresh=request.refresh,
        )
    except EvidenceFactError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The payment cannot form a valid evidence case.",
        ) from error
    except EvidenceGuardrailError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The evidence request failed safety validation.",
        ) from error
    except (AuditIntegrityError, EvidenceStoreError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The evidence subsystem failed closed.",
        ) from error
