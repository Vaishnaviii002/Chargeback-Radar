from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.evidence_contract import (
    EvidenceCase,
    EvidenceFact,
    FactSource,
)
from src.evidence_generator import (
    EvidenceGenerationError,
    EvidenceGeneratorConfig,
    generate_evidence_pack,
)


UTC = timezone.utc
OUTPUT_PATH = Path("reports/evidence_smoke.json")


def build_smoke_case() -> EvidenceCase:
    """Return a safe synthetic case for the first real API request."""
    payment_created_at = datetime(2026, 1, 1, tzinfo=UTC)
    as_of = datetime(2026, 1, 20, tzinfo=UTC)

    return EvidenceCase(
        payment_id="pay_smoke_000001",
        as_of=as_of,
        deterministic_recommended_action="PREPARE_EVIDENCE",
        calibrated_probability=0.18,
        risk_band="CRITICAL",
        triggered_rule_codes=["SHIPMENT_SLA_BREACHED"],
        facts=[
            EvidenceFact(
                fact_id="PAYMENT_CREATED_AT",
                source=FactSource.PAYMENT,
                label="Payment created at",
                value=payment_created_at.isoformat(),
                observed_at=payment_created_at,
            ),
            EvidenceFact(
                fact_id="PAYMENT_AMOUNT",
                source=FactSource.PAYMENT,
                label="Payment amount",
                value="INR 2,500.00",
                observed_at=payment_created_at,
            ),
            EvidenceFact(
                fact_id="THREEDS_STATUS",
                source=FactSource.PAYMENT,
                label="3DS status",
                value="authenticated",
                observed_at=payment_created_at,
            ),
            EvidenceFact(
                fact_id="SHIPMENT_PROMISED_AT",
                source=FactSource.SHIPMENT,
                label="Promised shipment deadline",
                value="2026-01-10T00:00:00+00:00",
                observed_at=payment_created_at,
            ),
            EvidenceFact(
                fact_id="MODEL_CALIBRATED_PROBABILITY",
                source=FactSource.MODEL,
                label="Calibrated chargeback probability",
                value="18.00%",
                observed_at=payment_created_at,
            ),
            EvidenceFact(
                fact_id="POLICY_RECOMMENDED_ACTION",
                source=FactSource.POLICY,
                label="Lowest expected-cost policy action",
                value="PREPARE_EVIDENCE",
                observed_at=payment_created_at,
            ),
            EvidenceFact(
                fact_id="LIFECYCLE_SHIPMENT_SLA_BREACHED",
                source=FactSource.LIFECYCLE_RULE,
                label="Triggered rule: SHIPMENT_SLA_BREACHED",
                value=(
                    "The promised shipment deadline passed without an "
                    "on-time delivery visible as of 2026-01-20."
                ),
                observed_at=as_of,
            ),
        ],
    )


def write_result(result_json: str) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = OUTPUT_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(result_json + "\n", encoding="utf-8")
    temporary_path.replace(OUTPUT_PATH)


def main() -> None:
    config = EvidenceGeneratorConfig.from_env()
    evidence_case = build_smoke_case()

    print("Chargeback Radar - Live OpenAI Evidence Smoke Test")
    print("-------------------------------------------------")
    print(f"Model configured: {config.model}")
    print(f"API key detected: {'yes' if config.api_key else 'no'}")
    print(f"Allowlisted facts sent: {len(evidence_case.facts)}")
    print(
        "Locked action: "
        f"{evidence_case.deterministic_recommended_action}"
    )

    try:
        result = generate_evidence_pack(
            evidence_case,
            config=config,
        )
    except EvidenceGenerationError as error:
        print(f"\nFAILED: {error}")
        raise SystemExit(1) from error

    write_result(result.model_dump_json(indent=2))

    cited_fact_ids = {
        fact_id
        for item in result.evidence_pack.evidence_items
        for fact_id in item.citation_fact_ids
    }

    print(f"Response ID received: {'yes' if result.response_id else 'no'}")
    print(f"Validated citations: {len(cited_fact_ids)}")
    print(f"Latency: {result.latency_ms} ms")
    print(f"Saved: {OUTPUT_PATH.as_posix()}")
    print("\nPASSED: structured output and business invariants validated.")
    print("No refund, message, or financial action was executed.")


if __name__ == "__main__":
    main()
