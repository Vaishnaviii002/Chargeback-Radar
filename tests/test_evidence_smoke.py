from __future__ import annotations

import json

from src.evidence_smoke import build_smoke_case, write_result


def test_smoke_case_is_safe_and_action_locked() -> None:
    case = build_smoke_case()
    serialized = case.model_dump_json()

    assert case.payment_id == "pay_smoke_000001"
    assert case.deterministic_recommended_action == "PREPARE_EVIDENCE"
    assert case.triggered_rule_codes == ["SHIPMENT_SLA_BREACHED"]
    assert len(case.facts) == 7
    assert "customer_id" not in serialized
    assert "chargeback_within_120d" not in serialized
    assert "reason_code" not in serialized


def test_result_write_is_atomic(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "reports" / "evidence_smoke.json"
    monkeypatch.setattr(
        "src.evidence_smoke.OUTPUT_PATH",
        output_path,
    )

    write_result('{"status":"validated"}')

    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "status": "validated"
    }
    assert not output_path.with_suffix(".json.tmp").exists()
