import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
)


REPORTS_DIR = Path("reports")
TEST_PATH = REPORTS_DIR / "test_scored.parquet"

TARGET = "chargeback_within_120d"

ACTIONS = [
    "MONITOR",
    "PREPARE_EVIDENCE",
    "MANUAL_REVIEW",
    "RECOMMEND_REFUND",
]


@dataclass
class CostParams:
    chargeback_fee: float = 1_500.0
    gross_margin_rate: float = 0.35
    evidence_cost: float = 40.0
    manual_review_cost: float = 150.0
    customer_friction_rate: float = 0.08
    evidence_recovery_rate: float = 0.45
    review_prevention_rate: float = 0.65
    refund_cost_rate: float = 0.35
    risk_program_penalty: float = 800.0


def calculate_expected_costs(
    probabilities: np.ndarray,
    amounts: np.ndarray,
    params: CostParams,
) -> np.ndarray:
    total_chargeback_loss = (
        amounts
        + params.chargeback_fee
        + params.risk_program_penalty
    )

    monitor_cost = (
        probabilities * total_chargeback_loss
    )

    prepare_evidence_cost = (
        params.evidence_cost
        + probabilities
        * (
            total_chargeback_loss
            - params.evidence_recovery_rate * amounts
        )
    )

    manual_review_cost = (
        params.manual_review_cost
        + probabilities
        * (1 - params.review_prevention_rate)
        * total_chargeback_loss
        + (1 - probabilities)
        * params.customer_friction_rate
        * amounts
        * params.gross_margin_rate
    )

    recommend_refund_cost = (
        amounts * params.refund_cost_rate
    )

    return np.column_stack(
        [
            monitor_cost,
            prepare_evidence_cost,
            manual_review_cost,
            recommend_refund_cost,
        ]
    )


def calculate_realised_costs(
    actions: np.ndarray,
    labels: np.ndarray,
    amounts: np.ndarray,
    params: CostParams,
) -> np.ndarray:
    residual_loss, intervention_cost = (
        calculate_realised_cost_components(
            actions,
            labels,
            amounts,
            params,
        )
    )

    return residual_loss + intervention_cost


def calculate_realised_cost_components(
    actions: np.ndarray,
    labels: np.ndarray,
    amounts: np.ndarray,
    params: CostParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate remaining loss from the cost of intervening."""
    total_chargeback_loss = (
        amounts
        + params.chargeback_fee
        + params.risk_program_penalty
    )

    residual_loss = np.zeros(len(actions), dtype=float)
    intervention_cost = np.zeros(len(actions), dtype=float)

    monitor_mask = actions == "MONITOR"
    evidence_mask = actions == "PREPARE_EVIDENCE"
    review_mask = actions == "MANUAL_REVIEW"
    refund_mask = actions == "RECOMMEND_REFUND"

    residual_loss[monitor_mask] = (
        labels[monitor_mask]
        * total_chargeback_loss[monitor_mask]
    )

    residual_loss[evidence_mask] = (
        labels[evidence_mask]
        * (
            total_chargeback_loss[evidence_mask]
            - params.evidence_recovery_rate
            * amounts[evidence_mask]
        )
    )
    intervention_cost[evidence_mask] = params.evidence_cost

    residual_loss[review_mask] = (
        labels[review_mask]
        * (1 - params.review_prevention_rate)
        * total_chargeback_loss[review_mask]
    )
    intervention_cost[review_mask] = (
        params.manual_review_cost
        + (1 - labels[review_mask])
        * params.customer_friction_rate
        * amounts[review_mask]
        * params.gross_margin_rate
    )

    intervention_cost[refund_mask] = (
        amounts[refund_mask]
        * params.refund_cost_rate
    )

    return residual_loss, intervention_cost


def simulate_policy(
    test_data: pd.DataFrame,
    params: CostParams,
) -> tuple[dict, pd.DataFrame]:
    probabilities = test_data[
        "calibrated_probability"
    ].to_numpy(dtype=float)

    labels = test_data[TARGET].to_numpy(dtype=int)

    amounts = (
        test_data["amount_paise"].to_numpy(dtype=float) / 100
    )

    expected_costs = calculate_expected_costs(
        probabilities,
        amounts,
        params,
    )

    chosen_action_indexes = np.argmin(
        expected_costs,
        axis=1,
    )

    chosen_actions = np.array(ACTIONS)[
        chosen_action_indexes
    ]

    residual_chargeback_loss, intervention_cost = (
        calculate_realised_cost_components(
            chosen_actions,
            labels,
            amounts,
            params,
        )
    )

    realised_policy_cost = (
        residual_chargeback_loss + intervention_cost
    )

    total_chargeback_loss = (
        amounts
        + params.chargeback_fee
        + params.risk_program_penalty
    )

    baseline_cost_per_transaction = (
        labels * total_chargeback_loss
    )

    baseline_cost = float(
        baseline_cost_per_transaction.sum()
    )

    remaining_chargeback_loss = float(
        residual_chargeback_loss.sum()
    )

    total_intervention_cost = float(
        intervention_cost.sum()
    )

    gross_avoided_loss = (
        baseline_cost - remaining_chargeback_loss
    )

    policy_cost = float(realised_policy_cost.sum())
    estimated_net_benefit = baseline_cost - policy_cost

    intervention_mask = chosen_actions != "MONITOR"
    false_positive_mask = (
        (labels == 0) & intervention_mask
    )

    binary_intervention = intervention_mask.astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        binary_intervention,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        labels,
        binary_intervention,
        zero_division=0,
    )

    recall = recall_score(
        labels,
        binary_intervention,
        zero_division=0,
    )

    false_positive_cost = float(
        realised_policy_cost[false_positive_mask].sum()
    )

    action_mix = {
        action: int((chosen_actions == action).sum())
        for action in ACTIONS
    }

    scored_data = test_data.copy()
    scored_data["recommended_action"] = chosen_actions
    scored_data["realised_policy_cost"] = realised_policy_cost
    scored_data["residual_chargeback_loss"] = (
        residual_chargeback_loss
    )
    scored_data["intervention_cost"] = intervention_cost
    scored_data["baseline_cost"] = (
        baseline_cost_per_transaction
    )

    for index, action in enumerate(ACTIONS):
        column_name = (
            f"expected_cost_{action.lower()}"
        )

        scored_data[column_name] = expected_costs[:, index]

    results = {
        "parameters": asdict(params),
        "records_evaluated": len(test_data),
        "action_mix": action_mix,
        "intervention_count": int(intervention_mask.sum()),
        "intervention_rate": float(
            intervention_mask.mean()
        ),
        "precision": float(precision),
        "recall": float(recall),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "costs": {
            "do_nothing_baseline_rupees": baseline_cost,
            "remaining_chargeback_loss_rupees": (
                remaining_chargeback_loss
            ),
            "gross_avoided_loss_rupees": gross_avoided_loss,
            "intervention_cost_rupees": (
                total_intervention_cost
            ),
            "policy_cost_rupees": policy_cost,
            "false_positive_cost_rupees": (
                false_positive_cost
            ),
            "estimated_net_benefit_rupees": (
                estimated_net_benefit
            ),
        },
        "disclosure": (
            "This is a held-out synthetic policy backtest. "
            "Net benefit depends on the displayed intervention "
            "effectiveness and cost assumptions."
        ),
    }

    return results, scored_data


def main() -> None:
    if not TEST_PATH.exists():
        raise FileNotFoundError(
            "reports/test_scored.parquet is missing. "
            "Run calibration first."
        )

    test_data = pd.read_parquet(TEST_PATH)
    params = CostParams()

    results, scored_data = simulate_policy(
        test_data,
        params,
    )

    with open(
        REPORTS_DIR / "policy_default.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=2)

    scored_data.to_parquet(
        REPORTS_DIR / "test_policy.parquet",
        index=False,
    )

    print("\nChargeback Radar — Default Policy Backtest")
    print("-" * 58)
    print(
        f"Records evaluated:       "
        f"{results['records_evaluated']:,}"
    )
    print(
        f"Intervention rate:       "
        f"{results['intervention_rate']:.3%}"
    )
    print(
        f"Policy precision:        "
        f"{results['precision']:.3%}"
    )
    print(
        f"Policy recall:           "
        f"{results['recall']:.3%}"
    )

    print("\nAction mix:")
    for action, count in results["action_mix"].items():
        print(f"{action:<22} {count:>8,}")

    print("\nCost backtest:")
    costs = results["costs"]

    print(
        "Do-nothing baseline:     "
        f"₹{costs['do_nothing_baseline_rupees']:,.2f}"
    )
    print(
        "Policy cost:             "
        f"₹{costs['policy_cost_rupees']:,.2f}"
    )
    print(
        "Gross avoided loss:      "
        f"₹{costs['gross_avoided_loss_rupees']:,.2f}"
    )
    print(
        "Intervention cost:       "
        f"₹{costs['intervention_cost_rupees']:,.2f}"
    )
    print(
        "False-positive cost:     "
        f"₹{costs['false_positive_cost_rupees']:,.2f}"
    )
    print(
        "Estimated net benefit:   "
        f"₹{costs['estimated_net_benefit_rupees']:,.2f}"
    )

    print("\nFiles created successfully:")
    print("  reports/policy_default.json")
    print("  reports/test_policy.parquet")


if __name__ == "__main__":
    main()