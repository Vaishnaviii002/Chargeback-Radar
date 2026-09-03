from __future__ import annotations

import numpy as np
import pandas as pd

from src.decide import CostParams, simulate_policy
from src.effectiveness_sensitivity import (
    build_effectiveness_sensitivity,
    build_scenario_parameters,
)


def make_test_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "payment_id": [
                "pay_low",
                "pay_medium",
                "pay_high",
                "pay_certain",
            ],
            "amount_paise": [
                100_000,
                250_000,
                500_000,
                1_000_000,
            ],
            "calibrated_probability": [
                0.001,
                0.05,
                0.35,
                0.90,
            ],
            "chargeback_within_120d": [0, 0, 1, 1],
        }
    )


def test_creates_low_base_high_scenarios() -> None:
    report = build_effectiveness_sensitivity(
        make_test_data()
    )

    assert report["scenario_order"] == [
        "LOW",
        "BASE",
        "HIGH",
    ]
    assert [
        scenario["scenario"]
        for scenario in report["scenarios"]
    ] == ["LOW", "BASE", "HIGH"]


def test_effectiveness_assumptions_are_ordered() -> None:
    scenarios = build_scenario_parameters(CostParams())

    evidence_rates = [
        params.evidence_recovery_rate
        for _, params in scenarios
    ]
    review_rates = [
        params.review_prevention_rate
        for _, params in scenarios
    ]

    assert evidence_rates == sorted(evidence_rates)
    assert review_rates == sorted(review_rates)


def test_base_scenario_matches_default_policy() -> None:
    test_data = make_test_data()
    params = CostParams()

    report = build_effectiveness_sensitivity(
        test_data,
        params,
    )
    default_result, _ = simulate_policy(test_data, params)
    base = report["scenarios"][1]

    assert base["action_mix"] == default_result["action_mix"]
    assert np.isclose(
        base["costs"]["estimated_net_benefit_rupees"],
        default_result["costs"][
            "estimated_net_benefit_rupees"
        ],
    )


def test_report_contains_complete_cost_breakdown() -> None:
    report = build_effectiveness_sensitivity(
        make_test_data()
    )

    required_costs = {
        "do_nothing_baseline_rupees",
        "remaining_chargeback_loss_rupees",
        "gross_avoided_loss_rupees",
        "intervention_cost_rupees",
        "policy_cost_rupees",
        "false_positive_cost_rupees",
        "estimated_net_benefit_rupees",
    }

    for scenario in report["scenarios"]:
        assert required_costs <= set(scenario["costs"])


def test_reported_ranges_match_scenario_values() -> None:
    report = build_effectiveness_sensitivity(
        make_test_data()
    )

    values = [
        scenario["costs"][
            "estimated_net_benefit_rupees"
        ]
        for scenario in report["scenarios"]
    ]
    reported_range = report["ranges"][
        "estimated_net_benefit_rupees"
    ]

    assert reported_range["minimum"] == min(values)
    assert reported_range["maximum"] == max(values)


def test_empty_test_data_is_rejected() -> None:
    try:
        build_effectiveness_sensitivity(pd.DataFrame())
    except ValueError as error:
        assert "requires test records" in str(error)
    else:
        raise AssertionError("Expected empty data to be rejected.")
