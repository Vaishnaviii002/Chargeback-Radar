from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = Path("README.md")
MANIFEST_PATH = Path("reports/submission_manifest.json")
START_MARKER = "<!-- CHARGEBACK_RADAR_METRICS:START -->"
END_MARKER = "<!-- CHARGEBACK_RADAR_METRICS:END -->"


def percent(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def rupees(value: float) -> str:
    return f"₹{value:,.0f}"


def render_metrics_block(manifest: dict[str, Any]) -> str:
    dataset = manifest["dataset"]
    evaluation = manifest["held_out_evaluation"]
    performance = evaluation["model_performance"]
    calibration = manifest["calibration"]
    policy = manifest["default_policy"]
    costs = policy["costs"]
    sensitivity_range = manifest["effectiveness_sensitivity"][
        "ranges"
    ]["estimated_net_benefit_rupees"]

    average_precision = performance["average_precision"]
    base_rate = evaluation["dataset"]["base_rate"]
    lift = average_precision / base_rate if base_rate else 0.0

    rows = [
        ("Synthetic payments", f"{dataset['payments']:,}"),
        ("Synthetic chargebacks", f"{dataset['chargebacks']:,}"),
        (
            "Overall base rate",
            percent(dataset["chargeback_rate"], 3),
        ),
        (
            "Held-out payments",
            f"{evaluation['dataset']['test_rows']:,}",
        ),
        (
            "Held-out chargebacks",
            f"{evaluation['dataset']['positive_labels']:,}",
        ),
        ("Average Precision", percent(average_precision)),
        ("Random/base-rate baseline", percent(base_rate, 3)),
        ("AP lift over baseline", f"{lift:.1f}×"),
        ("Precision", percent(performance["precision"])),
        ("Recall", percent(performance["recall"])),
        ("F1", f"{performance['f1']:.4f}"),
        (
            "Brier score before calibration",
            f"{calibration['test_raw_brier']:.6f}",
        ),
        (
            "Brier score after calibration",
            f"{calibration['test_calibrated_brier']:.6f}",
        ),
        (
            "False-positive count",
            f"{evaluation['confusion_matrix']['false_positive']:,}",
        ),
        (
            "Detector false-positive cost",
            rupees(
                evaluation["false_positive_cost"]["total_rupees"]
            ),
        ),
        (
            "Policy gross avoided loss",
            rupees(costs["gross_avoided_loss_rupees"]),
        ),
        (
            "Policy intervention cost",
            rupees(costs["intervention_cost_rupees"]),
        ),
        (
            "Policy estimated net benefit",
            rupees(costs["estimated_net_benefit_rupees"]),
        ),
        (
            "Effectiveness sensitivity range",
            (
                f"{rupees(sensitivity_range['minimum'])} to "
                f"{rupees(sensitivity_range['maximum'])}"
            ),
        ),
    ]

    table_lines = [
        "| Measure | Generated result |",
        "|---|---:|",
        *[
            f"| {label} | {value} |"
            for label, value in rows
        ],
    ]

    return "\n".join(
        [
            START_MARKER,
            "\n".join(table_lines),
            "",
            (
                "_Generated from the fixed-seed pipeline; monetary "
                "values are synthetic held-out backtest estimates._"
            ),
            END_MARKER,
        ]
    )


def update_readme_metrics(root: Path = REPO_ROOT) -> Path:
    readme_path = root / README_PATH
    manifest_path = root / MANIFEST_PATH

    if not readme_path.exists():
        raise FileNotFoundError("README.md is missing.")

    if not manifest_path.exists():
        raise FileNotFoundError(
            "reports/submission_manifest.json is missing. "
            "Run python -m src.pipeline first."
        )

    readme = readme_path.read_text(encoding="utf-8")
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    if START_MARKER not in readme or END_MARKER not in readme:
        raise ValueError("README metric markers are missing.")

    prefix, remainder = readme.split(START_MARKER, maxsplit=1)
    _, suffix = remainder.split(END_MARKER, maxsplit=1)

    updated = prefix + render_metrics_block(manifest) + suffix
    readme_path.write_text(updated, encoding="utf-8")

    return readme_path


def main() -> None:
    path = update_readme_metrics()
    print(f"README metrics refreshed from {MANIFEST_PATH}")
    print(f"Updated: {path}")


if __name__ == "__main__":
    main()
