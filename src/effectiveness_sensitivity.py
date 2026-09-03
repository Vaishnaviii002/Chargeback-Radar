

import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from src.decide import CostParams, simulate_policy


REPORTS_DIR = Path("reports")
TEST_PATH = REPORTS_DIR / "test_scored.parquet"
OUTPUT_PATH = REPORTS_DIR / "effectiveness_sensitivity.json"


def build_scenario_parameters(
    base_params: CostParams,
) -> list[tuple[str, CostParams]]:
    """Create conservative, declared effectiveness scenarios."""
    low = replace(
        base_params,
        evidence_recovery_rate=max(
            0.0,
            base_params.evidence_recovery_rate - 0.20,
        ),
        review_prevention_rate=max(
            0.0,
            base_params.review_prevention_rate - 0.20,
        ),
    )

    high = replace(
        base_params,
        evidence_recovery_rate=min(
            1.0,
            base_params.evidence_recovery_rate + 0.20,
        ),
        review_prevention_rate=min(
            1.0,
            base_params.review_prevention_rate + 0.15,
        ),
    )

    return [
        ("LOW", low),
        ("BASE", replace(base_params)),
        ("HIGH", high),
    ]


def build_effectiveness_sensitivity(
    test_data: pd.DataFrame,
    base_params: CostParams | None = None,
) -> dict:
    """Backtest the policy under three effectiveness assumptions."""
    if test_data.empty:
        raise ValueError("Sensitivity analysis requires test records.")

    params = base_params or CostParams()
    scenarios = []

    for scenario_name, scenario_params in build_scenario_parameters(
        params
    ):
        result, _ = simulate_policy(
            test_data,
            scenario_params,
        )

        scenarios.append(
            {
                "scenario": scenario_name,
                "assumptions": {
                    "evidence_recovery_rate": (
                        scenario_params.evidence_recovery_rate
                    ),
                    "review_prevention_rate": (
                        scenario_params.review_prevention_rate
                    ),
                },
                "intervention_count": result[
                    "intervention_count"
                ],
                "intervention_rate": result[
                    "intervention_rate"
                ],
                "precision": result["precision"],
                "recall": result["recall"],
                "action_mix": result["action_mix"],
                "costs": result["costs"],
            }
        )

    net_benefits = [
        scenario["costs"][
            "estimated_net_benefit_rupees"
        ]
        for scenario in scenarios
    ]
    gross_avoided_losses = [
        scenario["costs"]["gross_avoided_loss_rupees"]
        for scenario in scenarios
    ]
    recalls = [
        scenario["recall"]
        for scenario in scenarios
    ]
    intervention_rates = [
        scenario["intervention_rate"]
        for scenario in scenarios
    ]

    return {
        "records_evaluated": len(test_data),
        "scenario_order": ["LOW", "BASE", "HIGH"],
        "base_parameters": asdict(params),
        "scenarios": scenarios,
        "ranges": {
            "estimated_net_benefit_rupees": {
                "minimum": min(net_benefits),
                "maximum": max(net_benefits),
            },
            "gross_avoided_loss_rupees": {
                "minimum": min(gross_avoided_losses),
                "maximum": max(gross_avoided_losses),
            },
            "recall": {
                "minimum": min(recalls),
                "maximum": max(recalls),
            },
            "intervention_rate": {
                "minimum": min(intervention_rates),
                "maximum": max(intervention_rates),
            },
        },
        "disclosure": (
            "LOW, BASE, and HIGH are assumption scenarios, not "
            "confidence intervals. Evidence recovery and manual-review "
            "prevention rates vary while all other costs remain fixed. "
            "All results use the same held-out synthetic test set."
        ),
    }


def main() -> None:
    if not TEST_PATH.exists():
        raise FileNotFoundError(
            "reports/test_scored.parquet is missing. "
            "Run calibration before sensitivity analysis."
        )

    test_data = pd.read_parquet(TEST_PATH)
    report = build_effectiveness_sensitivity(test_data)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nChargeback Radar — Effectiveness Sensitivity")
    print("-" * 61)

    for scenario in report["scenarios"]:
        assumptions = scenario["assumptions"]
        costs = scenario["costs"]

        print(
            f"{scenario['scenario']:<5}  "
            f"evidence={assumptions['evidence_recovery_rate']:.0%}  "
            f"review={assumptions['review_prevention_rate']:.0%}  "
            f"recall={scenario['recall']:.1%}  "
            "net benefit="
            f"₹{costs['estimated_net_benefit_rupees']:,.2f}"
        )

    benefit_range = report["ranges"][
        "estimated_net_benefit_rupees"
    ]

    print("\nEstimated net-benefit sensitivity range:")
    print(
        f"₹{benefit_range['minimum']:,.2f} "
        f"to ₹{benefit_range['maximum']:,.2f}"
    )
    print("\nFile created successfully:")
    print("  reports/effectiveness_sensitivity.json")


if __name__ == "__main__":
    main()