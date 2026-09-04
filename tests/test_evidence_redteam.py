from __future__ import annotations

import json

from src.evidence_redteam import (
    run_redteam,
    scenarios,
    write_report,
)


def test_redteam_has_clean_control_and_twelve_attacks() -> None:
    suite = scenarios()

    assert len(suite) == 13
    assert sum(item.expected == "ALLOW" for item in suite) == 1
    assert sum(item.expected == "BLOCK" for item in suite) == 12


def test_all_redteam_scenarios_pass() -> None:
    report = run_redteam()

    assert report.summary.total_scenarios == 13
    assert report.summary.passed_scenarios == 13
    assert report.summary.false_accepts == 0
    assert report.summary.false_rejects == 0
    assert report.summary.pass_rate == 1.0


def test_scenario_names_are_unique() -> None:
    names = [scenario.name for scenario in scenarios()]

    assert len(names) == len(set(names))


def test_report_covers_winning_submission_risk_categories() -> None:
    categories = {scenario.category for scenario in scenarios()}

    assert {
        "input_security",
        "privacy",
        "decision_integrity",
        "grounding",
        "evidence_quality",
        "responsible_ai",
        "action_safety",
        "transparency",
    }.issubset(categories)


def test_written_report_contains_no_attack_pii(tmp_path) -> None:
    path = tmp_path / "evidence_guardrail_eval.json"
    write_report(run_redteam(), path)

    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)

    assert payload["summary"]["pass_rate"] == 1.0
    assert "customer@example.com" not in text
    assert "9876543210" not in text
    assert "4111 1111 1111 1111" not in text
