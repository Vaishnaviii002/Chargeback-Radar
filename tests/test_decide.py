import numpy as np
import pandas as pd

from src.decide import (
    ACTIONS,
    CostParams,
    calculate_expected_costs,
    simulate_policy,
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
            "chargeback_within_120d": [
                0,
                0,
                1,
                1,
            ],
        }
    )


def test_expected_cost_matrix_shape():
    probabilities = np.array([0.01, 0.20, 0.80])
    amounts = np.array([1_000.0, 2_000.0, 5_000.0])

    costs = calculate_expected_costs(
        probabilities,
        amounts,
        CostParams(),
    )

    assert costs.shape == (3, 4)


def test_all_expected_costs_are_finite_and_non_negative():
    probabilities = np.array([0.0, 0.10, 0.50, 1.0])
    amounts = np.array([500.0, 1_000.0, 5_000.0, 10_000.0])

    costs = calculate_expected_costs(
        probabilities,
        amounts,
        CostParams(),
    )

    assert np.isfinite(costs).all()
    assert (costs >= 0).all()


def test_low_risk_payment_is_monitored():
    probabilities = np.array([0.0001])
    amounts = np.array([1_000.0])

    costs = calculate_expected_costs(
        probabilities,
        amounts,
        CostParams(),
    )

    selected_action = ACTIONS[int(np.argmin(costs[0]))]

    assert selected_action == "MONITOR"


def test_simulation_returns_every_action_bucket():
    results, scored_data = simulate_policy(
        make_test_data(),
        CostParams(),
    )

    assert set(results["action_mix"]) == set(ACTIONS)
    assert len(scored_data) == 4
    assert scored_data["recommended_action"].isin(ACTIONS).all()


def test_simulation_returns_valid_metrics():
    results, _ = simulate_policy(
        make_test_data(),
        CostParams(),
    )

    assert 0 <= results["precision"] <= 1
    assert 0 <= results["recall"] <= 1
    assert 0 <= results["intervention_rate"] <= 1

    assert (
        results["costs"]["do_nothing_baseline_rupees"]
        >= 0
    )

    assert results["costs"]["policy_cost_rupees"] >= 0
    assert results["costs"]["false_positive_cost_rupees"] >= 0


def test_more_expensive_manual_review_does_not_increase_reviews():
    test_data = make_test_data()

    low_review_cost = CostParams(
        manual_review_cost=20.0
    )

    high_review_cost = CostParams(
        manual_review_cost=2_000.0
    )

    _, low_cost_scored = simulate_policy(
        test_data,
        low_review_cost,
    )

    _, high_cost_scored = simulate_policy(
        test_data,
        high_review_cost,
    )

    low_review_count = (
        low_cost_scored["recommended_action"]
        == "MANUAL_REVIEW"
    ).sum()

    high_review_count = (
        high_cost_scored["recommended_action"]
        == "MANUAL_REVIEW"
    ).sum()

    assert high_review_count <= low_review_count


def test_net_benefit_calculation_is_consistent():
    results, _ = simulate_policy(
        make_test_data(),
        CostParams(),
    )

    costs = results["costs"]

    expected_net_benefit = (
        costs["do_nothing_baseline_rupees"]
        - costs["policy_cost_rupees"]
    )

    assert np.isclose(
        costs["estimated_net_benefit_rupees"],
        expected_net_benefit,
    )