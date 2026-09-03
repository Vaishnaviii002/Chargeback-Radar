import json

import pandas as pd
import pytest

from src.decide import CostParams
from src.policy_lab import (
    build_policy_frontier,
    validate_review_costs,
)


def make_test_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "payment_id": [
                "pay_low",
                "pay_medium",
                "pay_high",
                "pay_critical",
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


def test_frontier_returns_every_cost_scenario():
    frontier = build_policy_frontier(
        make_test_data(),
        review_costs=[0, 50, 150, 400],
    )

    assert len(frontier["points"]) == 4

    returned_costs = [
        point["manual_review_cost_rupees"]
        for point in frontier["points"]
    ]

    assert returned_costs == [
        0,
        50,
        150,
        400,
    ]


def test_review_costs_are_sorted_and_deduplicated():
    values = validate_review_costs(
        [400, 50, 150, 50, 0]
    )

    assert values == [
        0,
        50,
        150,
        400,
    ]


def test_negative_review_cost_is_rejected():
    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        validate_review_costs(
            [0, 100, -50]
        )


def test_empty_review_cost_grid_is_rejected():
    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        validate_review_costs([])


def test_action_mix_matches_population_size():
    test_data = make_test_data()

    frontier = build_policy_frontier(
        test_data,
        review_costs=[0, 150, 600],
    )

    for point in frontier["points"]:
        assert (
            sum(point["action_mix"].values())
            == len(test_data)
        )


def test_more_expensive_review_does_not_increase_reviews():
    frontier = build_policy_frontier(
        make_test_data(),
        review_costs=[
            0,
            50,
            150,
            400,
            1_000,
        ],
    )

    review_counts = [
        point["action_mix"]["MANUAL_REVIEW"]
        for point in frontier["points"]
    ]

    assert review_counts == sorted(
        review_counts,
        reverse=True,
    )


def test_default_scenario_matches_default_cost():
    frontier = build_policy_frontier(
        make_test_data(),
        base_params=CostParams(
            manual_review_cost=150
        ),
        review_costs=[0, 50, 150, 400],
    )

    assert (
        frontier["default_scenario"][
            "manual_review_cost_rupees"
        ]
        == 150
    )


def test_frontier_is_json_serializable():
    frontier = build_policy_frontier(
        make_test_data(),
        review_costs=[0, 150, 400],
    )

    serialized = json.dumps(frontier)

    assert "points" in serialized
    assert "estimated_net_benefit_rupees" in serialized
    assert "disclosures" in serialized