from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path("reports/submission_manifest.json")

PIPELINE_STEPS = [
    ("Generate customers and payments", "src.generate"),
    ("Generate delayed chargeback outcomes", "src.outcomes"),
    ("Generate post-payment operations", "src.operations"),
    ("Build leakage-safe features", "src.features"),
    ("Train customer-disjoint model", "src.train"),
    ("Calibrate probabilities", "src.calibrate"),
    ("Evaluate untouched test set", "src.evaluate"),
    ("Backtest default policy", "src.decide"),
    ("Build review-cost frontier", "src.policy_lab"),
    (
        "Build effectiveness sensitivity",
        "src.effectiveness_sensitivity",
    ),
]

REQUIRED_OUTPUTS = [
    "data/customers.parquet",
    "data/raw_payments.parquet",
    "data/payments.parquet",
    "data/disputes.parquet",
    "data/operations.parquet",
    "data/features.parquet",
    "artifacts/feature_schema.json",
    "artifacts/model_bundle.joblib",
    "artifacts/model_metadata.json",
    "artifacts/calibrator.joblib",
    "reports/calibration_metrics.json",
    "reports/calibration_curve.json",
    "reports/metrics.json",
    "reports/precision_recall_curve.json",
    "reports/policy_default.json",
    "reports/policy_frontier.json",
    "reports/effectiveness_sensitivity.json",
    "reports/test_scored.parquet",
    "reports/test_policy.parquet",
]


@dataclass(frozen=True)
class StepResult:
    name: str
    module: str
    status: str
    duration_seconds: float


def read_json(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path

    if not path.exists():
        raise FileNotFoundError(
            f"Required report is missing: {relative_path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def git_provenance(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {
            "commit": None,
            "working_tree_dirty": None,
        }

    return {
        "commit": commit,
        "working_tree_dirty": bool(status.strip()),
    }


def validate_required_outputs(root: Path) -> None:
    missing = [
        relative_path
        for relative_path in REQUIRED_OUTPUTS
        if not (root / relative_path).exists()
    ]

    if missing:
        formatted = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Pipeline finished without required outputs:\n"
            f"  - {formatted}"
        )


def build_submission_manifest(
    root: Path,
    step_results: list[StepResult],
) -> dict[str, Any]:
    validate_required_outputs(root)

    payments = pd.read_parquet(
        root / "data/payments.parquet",
        columns=["payment_id", "chargeback_within_120d"],
    )
    disputes = pd.read_parquet(
        root / "data/disputes.parquet",
        columns=["payment_id"],
    )
    operations = pd.read_parquet(
        root / "data/operations.parquet",
        columns=[
            "payment_id",
            "shipment_sla_breached",
            "refund_requested_at",
            "promised_refund_not_processed",
        ],
    )

    metrics = read_json(root, "reports/metrics.json")
    calibration = read_json(
        root,
        "reports/calibration_metrics.json",
    )
    model = read_json(root, "artifacts/model_metadata.json")
    feature_schema = read_json(
        root,
        "artifacts/feature_schema.json",
    )
    policy = read_json(root, "reports/policy_default.json")
    frontier = read_json(root, "reports/policy_frontier.json")
    effectiveness = read_json(
        root,
        "reports/effectiveness_sensitivity.json",
    )

    artifact_inventory = []

    for relative_path in REQUIRED_OUTPUTS:
        path = root / relative_path
        artifact_inventory.append(
            {
                "path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    return {
        "schema_version": "1.0",
        "project": "Chargeback Radar",
        "submission_track": "Razorpay AI Buildathon Track 02",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "reproducibility": {
            "python_version": sys.version.split()[0],
            "command": "python -m src.pipeline",
            "source": git_provenance(root),
            "steps": [
                asdict(result)
                for result in step_results
            ],
        },
        "dataset": {
            "synthetic": True,
            "payments": len(payments),
            "unique_payment_ids": int(
                payments["payment_id"].nunique()
            ),
            "chargebacks": len(disputes),
            "chargeback_rate": float(
                payments["chargeback_within_120d"].mean()
            ),
            "operational_rows": len(operations),
            "shipment_sla_breaches": int(
                operations["shipment_sla_breached"].sum()
            ),
            "refund_requests": int(
                operations["refund_requested_at"].notna().sum()
            ),
            "refund_sla_breaches": int(
                operations[
                    "promised_refund_not_processed"
                ].sum()
            ),
        },
        "evaluation_integrity": {
            "split": metrics["disclosures"]["split"],
            "customer_disjoint": model["customer_disjoint"],
            "target": feature_schema["target"],
            "model_features": feature_schema["model_features"],
            "excluded_rule_fields": feature_schema[
                "excluded_rule_fields"
            ],
        },
        "held_out_evaluation": metrics,
        "calibration": calibration,
        "model": model,
        "default_policy": policy,
        "review_cost_frontier": frontier,
        "effectiveness_sensitivity": effectiveness,
        "artifact_inventory": artifact_inventory,
        "defense_only_controls": {
            "automatic_financial_actions": False,
            "human_approval_required": True,
            "post_payment_rules_isolated_from_model_features": True,
        },
    }


def run_step(
    root: Path,
    name: str,
    module: str,
) -> StepResult:
    print("\n" + "=" * 72)
    print(f"RUNNING: {name}")
    print(f"COMMAND: {sys.executable} -m {module}")
    print("=" * 72)

    started = time.perf_counter()

    try:
        subprocess.run(
            [sys.executable, "-m", module],
            cwd=root,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        duration = time.perf_counter() - started
        raise RuntimeError(
            f"Pipeline stopped at '{name}' after "
            f"{duration:.1f}s. Fix that module and rerun."
        ) from error

    duration = time.perf_counter() - started
    print(f"COMPLETED: {name} ({duration:.1f}s)")

    return StepResult(
        name=name,
        module=module,
        status="completed",
        duration_seconds=round(duration, 3),
    )


def run_pipeline(root: Path = REPO_ROOT) -> Path:
    started = time.perf_counter()
    step_results = []

    for name, module in PIPELINE_STEPS:
        step_results.append(run_step(root, name, module))

    manifest = build_submission_manifest(root, step_results)
    output_path = root / MANIFEST_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    duration = time.perf_counter() - started
    evaluation = manifest["held_out_evaluation"]
    policy = manifest["default_policy"]

    print("\n" + "=" * 72)
    print("CHARGEBACK RADAR PIPELINE COMPLETED")
    print("=" * 72)
    print(f"Total duration:       {duration:.1f}s")
    print(
        "Held-out AP:          "
        f"{evaluation['model_performance']['average_precision']:.2%}"
    )
    print(
        "Held-out recall:      "
        f"{evaluation['model_performance']['recall']:.2%}"
    )
    print(
        "Policy net benefit:   "
        "₹"
        f"{policy['costs']['estimated_net_benefit_rupees']:,.2f}"
    )
    print(f"Manifest:             {MANIFEST_PATH}")

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate every Chargeback Radar submission artifact."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root. Defaults to the current project.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.root.resolve())


if __name__ == "__main__":
    main()
