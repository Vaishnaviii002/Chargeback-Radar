from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.decide import (
    CostParams,
    simulate_policy,
)


REPORTS_DIR = Path("reports")
TEST_PATH = REPORTS_DIR / "test_scored.parquet"
FRONTIER_PATH = REPORTS_DIR / "policy_frontier.json"

DEFAULT_REVIEW_COST_GRID = [
    0,
    50,
    100,
    150,
    250,
    400,
    600,
    1_000,
]


def validate_review_costs(
    review_costs: Iterable[float],
) -> list[float]:
    values = [
        float(value)
        for value in review_costs
    ]

    if not values:
        raise ValueError(
            "At least one review cost is required."
        )

    if any(value < 0 for value in values):
        raise ValueError(
            "Review costs cannot be negative."
        )

    return sorted(set(values))


def build_policy_frontier(
    test_data: pd.DataFrame,
    base_params: CostParams | None = None,
    review_costs: Iterable[float] | None = None,
) -> dict:
    """
    Evaluate the same held-out test population under multiple
    manual-review costs.

    This exposes the operational trade-off instead of presenting
    one cherry-picked policy configuration.
    """

    params = base_params or CostParams()

    cost_grid = validate_review_costs(
        review_costs
        if review_costs is not None
        else DEFAULT_REVIEW_COST_GRID
    )

    points: list[dict] = []

    for manual_review_cost in cost_grid:
        scenario_params = replace(
            params,
            manual_review_cost=manual_review_cost,
        )

        result, _ = simulate_policy(
            test_data,
            scenario_params,
        )

        confusion = result["confusion_matrix"]
        costs = result["costs"]

        baseline_cost = costs[
            "do_nothing_baseline_rupees"
        ]

        net_benefit = costs[
            "estimated_net_benefit_rupees"
        ]

        if baseline_cost > 0:
            benefit_rate = (
                net_benefit / baseline_cost
            )
        else:
            benefit_rate = 0.0

        points.append(
            {
                "manual_review_cost_rupees": (
                    manual_review_cost
                ),
                "intervention_count": result[
                    "intervention_count"
                ],
                "intervention_rate": result[
                    "intervention_rate"
                ],
                "precision": result["precision"],
                "recall": result["recall"],
                "true_positive": confusion[
                    "true_positive"
                ],
                "false_positive": confusion[
                    "false_positive"
                ],
                "false_negative": confusion[
                    "false_negative"
                ],
                "action_mix": result[
                    "action_mix"
                ],
                "policy_cost_rupees": costs[
                    "policy_cost_rupees"
                ],
                "false_positive_cost_rupees": costs[
                    "false_positive_cost_rupees"
                ],
                "estimated_net_benefit_rupees": (
                    net_benefit
                ),
                "estimated_benefit_rate": (
                    benefit_rate
                ),
            }
        )

    highest_benefit_point = max(
        points,
        key=lambda point: point[
            "estimated_net_benefit_rupees"
        ],
    )

    default_point = min(
        points,
        key=lambda point: abs(
            point["manual_review_cost_rupees"]
            - params.manual_review_cost
        ),
    )

    return {
        "records_evaluated": len(test_data),
        "variable": "manual_review_cost_rupees",
        "base_parameters": asdict(params),
        "default_scenario": default_point,
        "highest_estimated_benefit_scenario": (
            highest_benefit_point
        ),
        "points": points,
        "disclosures": [
            (
                "All scenarios use the same untouched, "
                "customer-disjoint synthetic test population."
            ),
            (
                "The highest-benefit scenario is sensitivity "
                "analysis, not a guaranteed recommendation."
            ),
            (
                "Estimated benefit depends on the displayed "
                "cost and intervention-effectiveness assumptions."
            ),
        ],
    }


def main() -> None:
    if not TEST_PATH.exists():
        raise FileNotFoundError(
            "reports/test_scored.parquet is missing. "
            "Run: python -m src.calibrate"
        )

    test_data = pd.read_parquet(TEST_PATH)

    frontier = build_policy_frontier(
        test_data
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        FRONTIER_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            frontier,
            file,
            indent=2,
        )

    print(
        "\nChargeback Radar — Policy Cost Frontier"
    )
    print("-" * 76)
    print(
        f"{'Review cost':>12} "
        f"{'Intervene':>12} "
        f"{'Precision':>12} "
        f"{'Recall':>10} "
        f"{'Net benefit':>16}"
    )

    for point in frontier["points"]:
        print(
            f"₹{point['manual_review_cost_rupees']:>11,.0f} "
            f"{point['intervention_rate']:>11.2%} "
            f"{point['precision']:>11.2%} "
            f"{point['recall']:>9.2%} "
            f"₹{point['estimated_net_benefit_rupees']:>14,.0f}"
        )

    print(
        "\nFile created successfully:"
    )
    print(
        "  reports/policy_frontier.json"
    )


if __name__ == "__main__":
    main()