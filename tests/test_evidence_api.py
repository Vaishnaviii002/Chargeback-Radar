from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.evidence_api import (
    EvidencePaymentNotFoundError,
    get_evidence_repository,
    get_evidence_service,
    router,
)
from src.evidence_contract import EvidenceFact, FactSource
from src.evidence_fallback import generate_fallback_evidence_pack
from src.evidence_guardrails import validate_generated_evidence_pack
from src.evidence_service import EvidenceDeliveryResult
from src.evidence_store import (
    AuditIntegrityError,
    AuditVerification,
)


UTC = timezone.utc
AS_OF = datetime(2026, 1, 20, tzinfo=UTC)


def payment() -> dict:
    return {
        "payment_id": "pay_test_001",
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "amount_paise": 250_000,
        "card_network": "Visa",
        "product_category": "electronics",
        "is_digital_good": False,
        "descriptor_clarity_score": 0.80,
        "phone_verified": True,
        "email_verified": True,
        "account_age_days": 365,
        "total_prior_orders": 5,
        "prior_disputes_count": 0,
        "txns_last_1h": 0,
        "txns_last_24h": 1,
        "txns_last_7d": 3,
        "amount_last_24h_paise": 250_000,
        "device_is_new": False,
        "ip_country_matches_billing": True,
        "ip_is_proxy_or_vpn": False,
        "cvv_result": "match",
        "threeds_status": "authenticated",
        "threeds_liability_shift": True,
        "billing_shipping_distance_km": 10,
        "is_duplicate_payment": False,
        "cancelled_subscription_billed": False,
        "calibrated_probability": 0.18,
        "recommended_action": "PREPARE_EVIDENCE",
    }


def operation() -> dict:
    return {
        "payment_id": "pay_test_001",
        "shipment_expected": True,
        "shipment_promised_at": datetime(2026, 1, 10, tzinfo=UTC),
        "shipment_delivered_at": None,
        "refund_requested_at": None,
        "refund_promised_by": None,
        "refund_processed_at": None,
    }


class FakeRepository:
    def __init__(self, *, missing: bool = False) -> None:
        self.missing = missing

    def get_case_records(self, payment_id: str):
        if self.missing:
            raise EvidencePaymentNotFoundError("not found")
        return payment(), operation()


class FakeAudit:
    def __init__(self, *, broken: bool = False) -> None:
        self.broken = broken

    def verify(self):
        if self.broken:
            raise AuditIntegrityError("tampered")
        return AuditVerification(
            records_verified=2,
            last_event_hash="a" * 64,
            integrity_mode="HMAC_SHA256_CHAIN",
        )


class FakeService:
    def __init__(self, *, broken_audit: bool = False) -> None:
        self.audit = FakeAudit(broken=broken_audit)
        self.generator = SimpleNamespace(
            config=SimpleNamespace(
                enabled=True,
                api_key="must-never-appear-in-response",
                model="gpt-5.6",
            )
        )
        self.last_case = None
        self.last_refresh = None

    def generate(self, evidence_case, *, refresh=False):
        self.last_case = evidence_case
        self.last_refresh = refresh
        pack = generate_fallback_evidence_pack(evidence_case)
        return EvidenceDeliveryResult(
            evidence_pack=pack,
            delivery_mode="DETERMINISTIC_FALLBACK",
            provider="deterministic_fallback",
            model=None,
            response_id=None,
            latency_ms=0,
            cache_hit=False,
            fallback_used=True,
            fallback_reason="AI_DISABLED",
            guardrails=validate_generated_evidence_pack(
                pack,
                evidence_case,
            ),
        )


def client_with(repository=None, service=None):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_evidence_repository] = (
        lambda: repository or FakeRepository()
    )
    app.dependency_overrides[get_evidence_service] = (
        lambda: service or FakeService()
    )
    return TestClient(app)


def test_status_reports_capabilities_without_secret() -> None:
    response = client_with().get("/api/evidence/status")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["api_key_configured"] is True
    assert body["automatic_action_execution"] is False
    assert "must-never-appear" not in response.text


def test_generate_returns_validated_fallback_pack() -> None:
    response = client_with().post(
        "/api/evidence/pay_test_001/generate",
        json={"as_of": AS_OF.isoformat(), "refresh": False},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["delivery_mode"] == "DETERMINISTIC_FALLBACK"
    assert body["evidence_pack"]["payment_id"] == "pay_test_001"
    assert body["evidence_pack"]["action_executed"] is False
    assert body["evidence_pack"]["human_approval_required"] is True
    assert response.headers["X-Chargeback-Radar-Request-Id"]


def test_generate_passes_refresh_to_service() -> None:
    service = FakeService()
    response = client_with(service=service).post(
        "/api/evidence/pay_test_001/generate",
        json={"as_of": AS_OF.isoformat(), "refresh": True},
    )

    assert response.status_code == 200
    assert service.last_refresh is True


def test_operation_timeline_reaches_fact_compiler() -> None:
    service = FakeService()
    response = client_with(service=service).post(
        "/api/evidence/pay_test_001/generate",
        json={"as_of": AS_OF.isoformat()},
    )

    assert response.status_code == 200
    fact_ids = {fact.fact_id for fact in service.last_case.facts}
    assert "SHIPMENT_PROMISED_AT" in fact_ids


def test_missing_payment_returns_404() -> None:
    response = client_with(
        repository=FakeRepository(missing=True)
    ).post(
        "/api/evidence/pay_missing_001/generate",
        json={"as_of": AS_OF.isoformat()},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Payment not found."


def test_invalid_payment_id_returns_422() -> None:
    response = client_with().post(
        "/api/evidence/not-a-payment/generate",
        json={"as_of": AS_OF.isoformat()},
    )

    assert response.status_code == 422


def test_naive_as_of_is_rejected() -> None:
    response = client_with().post(
        "/api/evidence/pay_test_001/generate",
        json={"as_of": "2026-01-20T00:00:00"},
    )

    assert response.status_code == 422
    assert "timezone" in response.json()["detail"]


def test_unknown_request_fields_are_rejected() -> None:
    response = client_with().post(
        "/api/evidence/pay_test_001/generate",
        json={
            "as_of": AS_OF.isoformat(),
            "execute_refund": True,
        },
    )

    assert response.status_code == 422


def test_audit_verify_returns_integrity_proof() -> None:
    response = client_with().get("/api/evidence/audit/verify")

    assert response.status_code == 200
    assert response.json()["records_verified"] == 2
    assert response.json()["integrity_mode"] == "HMAC_SHA256_CHAIN"


def test_broken_audit_fails_closed() -> None:
    response = client_with(
        service=FakeService(broken_audit=True)
    ).get("/api/evidence/audit/verify")

    assert response.status_code == 503
    assert "integrity" in response.json()["detail"]


def test_openapi_contains_evidence_endpoints() -> None:
    response = client_with().get("/openapi.json")

    paths = response.json()["paths"]
    assert "/api/evidence/status" in paths
    assert "/api/evidence/audit/verify" in paths
    assert "/api/evidence/{payment_id}/generate" in paths
