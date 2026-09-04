from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.evidence_contract import (
    EvidenceCase,
    EvidencePack,
    validate_pack_against_case,
)
from src.evidence_guardrails import validate_generated_evidence_pack


UTC = timezone.utc
ZERO_HASH = "0" * 64
HASH_PATTERN = re.compile(r"^[a-f0-9]{64}$")
DEFAULT_GENERATION_VERSION = "evidence-v1"


class EvidenceStoreError(RuntimeError):
    pass


class AuditIntegrityError(EvidenceStoreError):
    pass


def _load_local_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except ImportError:
        pass


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise EvidenceStoreError("Timestamps must include a timezone.")
    return value.astimezone(UTC)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def evidence_case_fingerprint(evidence_case: EvidenceCase) -> str:
    return _sha256(
        _canonical_bytes(
            evidence_case.model_dump(
                mode="json",
                exclude_none=True,
            )
        )
    )


def evidence_pack_fingerprint(evidence_pack: EvidencePack) -> str:
    return _sha256(
        _canonical_bytes(
            evidence_pack.model_dump(
                mode="json",
                exclude_none=True,
            )
        )
    )


def payment_reference_hash(payment_id: str) -> str:
    return _sha256(payment_id.encode("utf-8"))


def build_evidence_cache_key(
    evidence_case: EvidenceCase,
    *,
    generation_version: str = DEFAULT_GENERATION_VERSION,
) -> str:
    material = {
        "case_fingerprint": evidence_case_fingerprint(evidence_case),
        "generation_version": generation_version,
    }
    return _sha256(_canonical_bytes(material))


class EvidenceCacheEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    cache_key: str = Field(pattern=r"^[a-f0-9]{64}$")
    case_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    generation_version: str = Field(min_length=1, max_length=100)
    provider: str = Field(min_length=1, max_length=80)
    model: str | None = Field(default=None, max_length=100)
    response_id: str | None = Field(default=None, max_length=200)
    created_at: datetime
    expires_at: datetime
    evidence_pack: EvidencePack


class FileEvidenceCache:
    """Validated, atomically-written cache for generated evidence packs."""

    def __init__(
        self,
        root: str | Path = "runtime/evidence_cache",
        *,
        ttl_seconds: int = 21_600,
    ) -> None:
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be positive")

        self.root = Path(root)
        self.ttl_seconds = ttl_seconds

    @classmethod
    def from_env(cls) -> "FileEvidenceCache":
        _load_local_env()

        return cls(
            root=os.getenv(
                "EVIDENCE_CACHE_DIR",
                "runtime/evidence_cache",
            ),
            ttl_seconds=int(
                os.getenv("EVIDENCE_CACHE_TTL_SECONDS", "21600")
            ),
        )

    def _path(self, cache_key: str) -> Path:
        if not HASH_PATTERN.fullmatch(cache_key):
            raise EvidenceStoreError("Invalid evidence cache key.")

        return self.root / f"{cache_key}.json"

    def put(
        self,
        evidence_case: EvidenceCase,
        evidence_pack: EvidencePack,
        *,
        provider: str,
        model: str | None = None,
        response_id: str | None = None,
        generation_version: str = DEFAULT_GENERATION_VERSION,
        ttl_seconds: int | None = None,
        now: datetime | None = None,
    ) -> EvidenceCacheEntry:
        validate_pack_against_case(evidence_pack, evidence_case)
        validate_generated_evidence_pack(evidence_pack, evidence_case)

        created_at = _as_utc(now or _utc_now())

        effective_ttl = (
            self.ttl_seconds
            if ttl_seconds is None
            else ttl_seconds
        )

        if effective_ttl < 1:
            raise ValueError("ttl_seconds must be positive")

        cache_key = build_evidence_cache_key(
            evidence_case,
            generation_version=generation_version,
        )

        entry = EvidenceCacheEntry(
            cache_key=cache_key,
            case_fingerprint=evidence_case_fingerprint(evidence_case),
            generation_version=generation_version,
            provider=provider,
            model=model,
            response_id=response_id,
            created_at=created_at,
            expires_at=created_at + timedelta(seconds=effective_ttl),
            evidence_pack=evidence_pack,
        )

        self.root.mkdir(parents=True, exist_ok=True)
        destination = self._path(cache_key)

        temporary = destination.with_name(
            f".{destination.name}.{uuid4().hex}.tmp"
        )

        try:
            with temporary.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                handle.write(entry.model_dump_json(indent=2))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

        return entry

    def get(
        self,
        evidence_case: EvidenceCase,
        *,
        generation_version: str = DEFAULT_GENERATION_VERSION,
        now: datetime | None = None,
    ) -> EvidenceCacheEntry | None:
        cache_key = build_evidence_cache_key(
            evidence_case,
            generation_version=generation_version,
        )

        path = self._path(cache_key)

        if not path.exists():
            return None

        try:
            entry = EvidenceCacheEntry.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception as error:
            raise EvidenceStoreError(
                "The evidence cache entry is unreadable or invalid."
            ) from error

        expected_fingerprint = evidence_case_fingerprint(evidence_case)

        if (
            entry.cache_key != cache_key
            or entry.case_fingerprint != expected_fingerprint
            or entry.generation_version != generation_version
        ):
            raise EvidenceStoreError(
                "The evidence cache entry does not match this case."
            )

        current_time = _as_utc(now or _utc_now())

        if _as_utc(entry.expires_at) <= current_time:
            return None

        validate_pack_against_case(
            entry.evidence_pack,
            evidence_case,
        )
        validate_generated_evidence_pack(
            entry.evidence_pack,
            evidence_case,
        )

        return entry


class EvidenceAuditRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    sequence: int = Field(ge=1)
    event_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    occurred_at: datetime

    event_type: Literal[
        "EVIDENCE_GENERATED",
        "CACHE_HIT",
        "CACHE_READ_FAILED",
        "CACHE_WRITE_FAILED",
        "FALLBACK_USED",
        "GENERATION_FAILED",
    ]

    status: Literal["SUCCESS", "FAILURE"]

    payment_reference_hash: str = Field(
        pattern=r"^[a-f0-9]{64}$"
    )
    case_fingerprint: str = Field(
        pattern=r"^[a-f0-9]{64}$"
    )
    pack_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )

    provider: str = Field(min_length=1, max_length=80)
    model: str | None = Field(default=None, max_length=100)
    response_id: str | None = Field(default=None, max_length=200)

    cache_hit: bool = False
    latency_ms: int | None = Field(default=None, ge=0)

    error_category: str | None = Field(
        default=None,
        pattern=r"^[A-Z][A-Z0-9_]{2,63}$",
    )

    integrity_mode: Literal[
        "SHA256_CHAIN",
        "HMAC_SHA256_CHAIN",
    ]

    previous_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    event_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class AuditVerification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: Literal[True] = True
    records_verified: int = Field(ge=0)
    last_event_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    integrity_mode: Literal[
        "SHA256_CHAIN",
        "HMAC_SHA256_CHAIN",
    ]


class AuditCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    records: int = Field(ge=1)
    last_event_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    integrity_mode: Literal[
        "SHA256_CHAIN",
        "HMAC_SHA256_CHAIN",
    ]

    checkpoint_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class EvidenceAuditLog:
    """Append-only audit log with chained integrity verification."""

    def __init__(
        self,
        path: str | Path = "runtime/evidence_audit.jsonl",
        *,
        hmac_key: str | bytes | None = None,
    ) -> None:
        self.path = Path(path)

        if isinstance(hmac_key, str):
            hmac_key = hmac_key.encode("utf-8")

        if hmac_key is not None and len(hmac_key) < 16:
            raise ValueError(
                "hmac_key must contain at least 16 bytes"
            )

        self._hmac_key = hmac_key
        self._lock = threading.RLock()

    @classmethod
    def from_env(cls) -> "EvidenceAuditLog":
        _load_local_env()

        return cls(
            path=os.getenv(
                "EVIDENCE_AUDIT_PATH",
                "runtime/evidence_audit.jsonl",
            ),
            hmac_key=(
                os.getenv("EVIDENCE_AUDIT_HMAC_KEY") or None
            ),
        )

    @property
    def checkpoint_path(self) -> Path:
        return self.path.with_name(
            self.path.name + ".head.json"
        )

    @property
    def integrity_mode(
        self,
    ) -> Literal["SHA256_CHAIN", "HMAC_SHA256_CHAIN"]:
        if self._hmac_key:
            return "HMAC_SHA256_CHAIN"

        return "SHA256_CHAIN"

    def _record_hash(self, material: dict[str, Any]) -> str:
        payload = _canonical_bytes(material)

        if self._hmac_key:
            return hmac.new(
                self._hmac_key,
                payload,
                hashlib.sha256,
            ).hexdigest()

        return _sha256(payload)

    def _write_checkpoint(
        self,
        *,
        records: int,
        last_event_hash: str,
    ) -> None:
        material = {
            "schema_version": "1.0",
            "records": records,
            "last_event_hash": last_event_hash,
            "integrity_mode": self.integrity_mode,
        }

        checkpoint = AuditCheckpoint.model_validate(
            {
                **material,
                "checkpoint_hash": self._record_hash(material),
            }
        )

        temporary = self.checkpoint_path.with_name(
            f".{self.checkpoint_path.name}.{uuid4().hex}.tmp"
        )

        try:
            with temporary.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                handle.write(
                    checkpoint.model_dump_json(indent=2)
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(temporary, self.checkpoint_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _verify_checkpoint(
        self,
        records: list[EvidenceAuditRecord],
    ) -> None:
        if not records:
            if self.checkpoint_path.exists():
                raise AuditIntegrityError(
                    "The audit checkpoint exists without audit records."
                )

            return

        if not self.checkpoint_path.exists():
            raise AuditIntegrityError(
                "The audit checkpoint is missing."
            )

        try:
            checkpoint = AuditCheckpoint.model_validate_json(
                self.checkpoint_path.read_text(encoding="utf-8")
            )
        except Exception as error:
            raise AuditIntegrityError(
                "The audit checkpoint is unreadable or malformed."
            ) from error

        material = checkpoint.model_dump(
            mode="json",
            exclude={"checkpoint_hash"},
        )

        expected_hash = self._record_hash(material)

        if not hmac.compare_digest(
            checkpoint.checkpoint_hash,
            expected_hash,
        ):
            raise AuditIntegrityError(
                "The audit checkpoint hash failed."
            )

        if checkpoint.integrity_mode != self.integrity_mode:
            raise AuditIntegrityError(
                "The audit checkpoint integrity mode failed."
            )

        if (
            checkpoint.records != len(records)
            or checkpoint.last_event_hash
            != records[-1].event_hash
        ):
            raise AuditIntegrityError(
                "The audit log does not match its checkpoint."
            )

    def _read_and_verify(
        self,
    ) -> list[EvidenceAuditRecord]:
        if not self.path.exists():
            return []

        records: list[EvidenceAuditRecord] = []
        expected_previous_hash = ZERO_HASH

        try:
            lines = self.path.read_text(
                encoding="utf-8"
            ).splitlines()

            for line_number, line in enumerate(
                lines,
                start=1,
            ):
                if not line.strip():
                    continue

                record = (
                    EvidenceAuditRecord.model_validate_json(line)
                )

                if record.sequence != len(records) + 1:
                    raise AuditIntegrityError(
                        f"Audit sequence failed at line "
                        f"{line_number}."
                    )

                if (
                    record.previous_hash
                    != expected_previous_hash
                ):
                    raise AuditIntegrityError(
                        f"Audit chain failed at line "
                        f"{line_number}."
                    )

                if (
                    record.integrity_mode
                    != self.integrity_mode
                ):
                    raise AuditIntegrityError(
                        f"Audit integrity mode failed at line "
                        f"{line_number}."
                    )

                material = record.model_dump(
                    mode="json",
                    exclude={"event_hash"},
                )

                expected_hash = self._record_hash(material)

                if not hmac.compare_digest(
                    record.event_hash,
                    expected_hash,
                ):
                    raise AuditIntegrityError(
                        f"Audit hash failed at line "
                        f"{line_number}."
                    )

                records.append(record)
                expected_previous_hash = record.event_hash

        except AuditIntegrityError:
            raise
        except Exception as error:
            raise AuditIntegrityError(
                "The evidence audit log is unreadable or malformed."
            ) from error

        self._verify_checkpoint(records)

        return records

    def verify(self) -> AuditVerification:
        with self._lock:
            records = self._read_and_verify()

            return AuditVerification(
                records_verified=len(records),
                last_event_hash=(
                    records[-1].event_hash
                    if records
                    else ZERO_HASH
                ),
                integrity_mode=self.integrity_mode,
            )

    def append(
        self,
        *,
        event_type: Literal[
            "EVIDENCE_GENERATED",
            "CACHE_HIT",
            "CACHE_READ_FAILED",
            "CACHE_WRITE_FAILED",
            "FALLBACK_USED",
            "GENERATION_FAILED",
        ],
        status: Literal["SUCCESS", "FAILURE"],
        evidence_case: EvidenceCase,
        provider: str,
        evidence_pack: EvidencePack | None = None,
        model: str | None = None,
        response_id: str | None = None,
        cache_hit: bool = False,
        latency_ms: int | None = None,
        error_category: str | None = None,
        occurred_at: datetime | None = None,
    ) -> EvidenceAuditRecord:
        if status == "SUCCESS" and evidence_pack is None:
            raise EvidenceStoreError(
                "Successful audit events require an evidence pack."
            )

        if status == "FAILURE" and not error_category:
            raise EvidenceStoreError(
                "Failed audit events require an error category."
            )

        if evidence_pack is not None:
            validate_pack_against_case(
                evidence_pack,
                evidence_case,
            )

        with self._lock:
            records = self._read_and_verify()

            previous_hash = (
                records[-1].event_hash
                if records
                else ZERO_HASH
            )

            material = {
                "schema_version": "1.0",
                "sequence": len(records) + 1,
                "event_id": uuid4().hex,
                "occurred_at": _as_utc(
                    occurred_at or _utc_now()
                ).isoformat().replace("+00:00", "Z"),
                "event_type": event_type,
                "status": status,
                "payment_reference_hash": (
                    payment_reference_hash(
                        evidence_case.payment_id
                    )
                ),
                "case_fingerprint": (
                    evidence_case_fingerprint(
                        evidence_case
                    )
                ),
                "pack_fingerprint": (
                    evidence_pack_fingerprint(
                        evidence_pack
                    )
                    if evidence_pack is not None
                    else None
                ),
                "provider": provider,
                "model": model,
                "response_id": response_id,
                "cache_hit": cache_hit,
                "latency_ms": latency_ms,
                "error_category": error_category,
                "integrity_mode": self.integrity_mode,
                "previous_hash": previous_hash,
            }

            event_hash = self._record_hash(material)

            record = EvidenceAuditRecord.model_validate(
                {
                    **material,
                    "event_hash": event_hash,
                }
            )

            self.path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            serialized = record.model_dump_json() + "\n"

            with self.path.open(
                "a",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())

            self._write_checkpoint(
                records=record.sequence,
                last_event_hash=record.event_hash,
            )

            return record