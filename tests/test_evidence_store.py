from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import pytest

from src.evidence_contract import EvidenceCase, EvidenceFact, FactSource
from src.evidence_fallback import generate_fallback_evidence_pack
from src.evidence_store import (
    AuditIntegrityError,
    EvidenceAuditLog,
    EvidenceStoreError,
    FileEvidenceCache,
    ZERO_HASH,
    build_evidence_cache_key,
    evidence_case_fingerprint,
)


UTC = timezone.utc
NOW = datetime(2026, 1, 20, tzinfo=UTC)


def make_case() -> EvidenceCase:
    return EvidenceCase(
        payment_id="pay_test_001",
        as_of=NOW,
        deterministic_recommended_action="PREPARE_EVIDENCE",
        calibrated_probability=0.18,
        risk_band="CRITICAL",
        triggered_rule_codes=["SHIPMENT_SLA_BREACHED"],
        facts=[
            EvidenceFact(
                fact_id="PAYMENT_AMOUNT",
                source=FactSource.PAYMENT,
                label="Payment amount",
                value="INR 2,500.00",
            ),
            EvidenceFact(
                fact_id="SHIPMENT_PROMISED_AT",
                source=FactSource.SHIPMENT,
                label="Promised shipment deadline",
                value="2026-01-10T00:00:00+00:00",
            ),
        ],
    )


def test_cache_key_is_stable_and_versioned() -> None:
    case = make_case()

    assert build_evidence_cache_key(case) == build_evidence_cache_key(case)
    assert build_evidence_cache_key(
        case,
        generation_version="evidence-v1",
    ) != build_evidence_cache_key(
        case,
        generation_version="evidence-v2",
    )


def test_cache_round_trip_revalidates_pack(tmp_path) -> None:
    case = make_case()
    pack = generate_fallback_evidence_pack(case)
    cache = FileEvidenceCache(tmp_path / "cache", ttl_seconds=60)

    written = cache.put(
        case,
        pack,
        provider="deterministic_fallback",
        now=NOW,
    )
    loaded = cache.get(case, now=NOW + timedelta(seconds=30))

    assert loaded is not None
    assert loaded.cache_key == written.cache_key
    assert loaded.evidence_pack == pack


def test_expired_cache_entry_is_not_returned(tmp_path) -> None:
    case = make_case()
    cache = FileEvidenceCache(tmp_path / "cache", ttl_seconds=10)
    cache.put(
        case,
        generate_fallback_evidence_pack(case),
        provider="deterministic_fallback",
        now=NOW,
    )

    assert cache.get(
        case,
        now=NOW + timedelta(seconds=10),
    ) is None


def test_corrupt_cache_entry_fails_closed(tmp_path) -> None:
    case = make_case()
    cache = FileEvidenceCache(tmp_path / "cache")
    key = build_evidence_cache_key(case)
    path = cache.root / f"{key}.json"
    path.parent.mkdir(parents=True)
    path.write_text("not-json", encoding="utf-8")

    with pytest.raises(EvidenceStoreError, match="unreadable"):
        cache.get(case, now=NOW)


def test_audit_log_uses_hashes_not_raw_case_data(tmp_path) -> None:
    case = make_case()
    pack = generate_fallback_evidence_pack(case)
    path = tmp_path / "audit.jsonl"
    audit = EvidenceAuditLog(path)

    record = audit.append(
        event_type="FALLBACK_USED",
        status="SUCCESS",
        evidence_case=case,
        evidence_pack=pack,
        provider="deterministic_fallback",
        occurred_at=NOW,
    )
    raw = path.read_text(encoding="utf-8")

    assert record.previous_hash == ZERO_HASH
    assert record.case_fingerprint == evidence_case_fingerprint(case)
    assert "pay_test_001" not in raw
    assert "INR 2,500.00" not in raw
    assert audit.verify().records_verified == 1


def test_multiple_audit_records_form_a_chain(tmp_path) -> None:
    case = make_case()
    pack = generate_fallback_evidence_pack(case)
    audit = EvidenceAuditLog(tmp_path / "audit.jsonl")

    first = audit.append(
        event_type="EVIDENCE_GENERATED",
        status="SUCCESS",
        evidence_case=case,
        evidence_pack=pack,
        provider="openai",
        model="gpt-5.6",
        response_id="resp_test_001",
        occurred_at=NOW,
    )
    second = audit.append(
        event_type="CACHE_HIT",
        status="SUCCESS",
        evidence_case=case,
        evidence_pack=pack,
        provider="openai",
        cache_hit=True,
        occurred_at=NOW + timedelta(seconds=1),
    )

    assert second.sequence == 2
    assert second.previous_hash == first.event_hash
    assert audit.verify().last_event_hash == second.event_hash


def test_modified_audit_record_is_detected(tmp_path) -> None:
    case = make_case()
    pack = generate_fallback_evidence_pack(case)
    path = tmp_path / "audit.jsonl"
    audit = EvidenceAuditLog(path)
    audit.append(
        event_type="FALLBACK_USED",
        status="SUCCESS",
        evidence_case=case,
        evidence_pack=pack,
        provider="deterministic_fallback",
        occurred_at=NOW,
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    data["provider"] = "edited"
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")

    with pytest.raises(AuditIntegrityError, match="hash failed"):
        audit.verify()


def test_hmac_log_rejects_the_wrong_key(tmp_path) -> None:
    case = make_case()
    pack = generate_fallback_evidence_pack(case)
    path = tmp_path / "audit.jsonl"
    EvidenceAuditLog(
        path,
        hmac_key="correct-secret-with-enough-bytes",
    ).append(
        event_type="FALLBACK_USED",
        status="SUCCESS",
        evidence_case=case,
        evidence_pack=pack,
        provider="deterministic_fallback",
        occurred_at=NOW,
    )

    with pytest.raises(AuditIntegrityError):
        EvidenceAuditLog(
            path,
            hmac_key="wrong-secret-with-enough-bytes",
        ).verify()


def test_truncated_audit_tail_is_detected_by_checkpoint(tmp_path) -> None:
    case = make_case()
    pack = generate_fallback_evidence_pack(case)
    path = tmp_path / "audit.jsonl"
    audit = EvidenceAuditLog(path)
    for offset in range(2):
        audit.append(
            event_type="FALLBACK_USED",
            status="SUCCESS",
            evidence_case=case,
            evidence_pack=pack,
            provider="deterministic_fallback",
            occurred_at=NOW + timedelta(seconds=offset),
        )

    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    path.write_text(first_line + "\n", encoding="utf-8")

    with pytest.raises(AuditIntegrityError, match="checkpoint"):
        audit.verify()


def test_failed_event_requires_only_a_safe_error_category(tmp_path) -> None:
    case = make_case()
    audit = EvidenceAuditLog(tmp_path / "audit.jsonl")

    record = audit.append(
        event_type="GENERATION_FAILED",
        status="FAILURE",
        evidence_case=case,
        provider="openai",
        model="gpt-5.6",
        error_category="API_TIMEOUT",
        occurred_at=NOW,
    )

    assert record.pack_fingerprint is None
    assert record.error_category == "API_TIMEOUT"
    assert audit.verify().records_verified == 1
