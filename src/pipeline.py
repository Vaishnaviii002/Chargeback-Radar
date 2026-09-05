from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
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
    ("Generate SHAP explanations", "src.explain"),
    (
        "Generate deterministic explanation text",
        "src.model_explanation_service",
    ),
    ("Generate timestamped support events", "src.support_events"),
    (
        "Generate sanitized support signals",
        "src.support_signal_report",
    ),
    ("Run controlled support-signal ablation", "src.ablation"),
    (
        "Run deterministic evidence guardrails",
        "src.evidence_redteam",
    ),
    ("Generate model card", "src.model_card"),
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
    "reports/test_explanations.parquet",
    "reports/test_model_explanations.parquet",
    "data/support_events.parquet",
    "data/support_scoring.parquet",
    "reports/support_signals.parquet",
    "reports/ablation.json",
    "reports/ablation_test_predictions.parquet",
    "reports/evidence_guardrail_eval.json",
    "reports/model_card.json",
    "MODEL_CARD.md",
]

OFFLINE_REPRODUCTION_ENV = {
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
    # Stable build time immediately after the synthetic 2025 period.
    "SOURCE_DATE_EPOCH": "1767225600",
    "EVIDENCE_AI_ENABLED": "false",
    "MODEL_EXPLANATION_AI_ENABLED": "false",
    "SUPPORT_SIGNAL_AI_ENABLED": "false",
    "RAZORPAY_INTEGRATION_ENABLED": "false",
    "RAZORPAY_WEBHOOK_ENABLED": "false",
}


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
    ablation = read_json(root, "reports/ablation.json")
    model_card = read_json(root, "reports/model_card.json")
    evidence_guardrails = read_json(
        root,
        "reports/evidence_guardrail_eval.json",
    )
    shap_explanations = pd.read_parquet(
        root / "reports/test_explanations.parquet",
        columns=["explanation_version"],
    )
    text_explanations = pd.read_parquet(
        root / "reports/test_model_explanations.parquet",
        columns=["text_explanation_version"],
    )

    shap_versions = sorted(
        shap_explanations["explanation_version"]
        .astype(str)
        .unique()
        .tolist()
    )
    text_versions = sorted(
        text_explanations["text_explanation_version"]
        .astype(str)
        .unique()
        .tolist()
    )

    if len(shap_versions) != 1 or len(text_versions) != 1:
        raise ValueError(
            "Explanation reports must each contain exactly one version."
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
        "versions": {
            "model": model["model_version"],
            "dataset": None,
            "shap_explanation": shap_versions[0],
            "text_explanation": text_versions[0],
            "evidence_schema": evidence_guardrails["schema_version"],
            "razorpay_normalization": "razorpay-normalizer-v1",
        },
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
        "support_signal_ablation": ablation,
        "model_card": {
            "version": model_card["model_card_version"],
            "path": "MODEL_CARD.md",
            "json_path": "reports/model_card.json",
        },
        "evidence_guardrails": {
            "report_path": "reports/evidence_guardrail_eval.json",
            "summary": evidence_guardrails["summary"],
        },
        "artifact_inventory": artifact_inventory,
        "defense_only_controls": {
            "automatic_financial_actions": False,
            "human_approval_required": True,
            "post_payment_rules_isolated_from_model_features": True,
        },
        "razorpay_test_mode": {
            "real_money_used": False,
            "automatic_financial_actions": False,
            "checkout_signature_verified_server_side": True,
            "demo_data_excluded_from_canonical_evaluation": True,
            "normalization_version": "razorpay-normalizer-v1",
        },
        "validation": {
            "status": "NOT_RUN_BY_PIPELINE",
            "release_command": "python scripts/verify_release.py",
            "backend_test_command": "python -m pytest -q",
            "frontend_build_command": "npm --prefix frontend run build",
        },
    }


def run_step(
    root: Path,
    name: str,
    module: str,
    environment_overrides: dict[str, str] | None = None,
) -> StepResult:
    print("\n" + "=" * 72)
    print(f"RUNNING: {name}")
    print(f"COMMAND: {sys.executable} -m {module}")
    print("=" * 72)

    started = time.perf_counter()

    try:
        environment = os.environ.copy()
        environment.update(OFFLINE_REPRODUCTION_ENV)
        environment.update(environment_overrides or {})
        subprocess.run(
            [sys.executable, "-m", module],
            cwd=root,
            check=True,
            env=environment,
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


def write_json_atomic(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            json.dump(payload, file, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
            temporary_path = Path(file.name)

        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def run_pipeline(root: Path = REPO_ROOT) -> Path:
    started = time.perf_counter()
    step_results = []

    with tempfile.TemporaryDirectory(
        prefix="chargeback-radar-pipeline-"
    ) as runtime_directory:
        isolated_runtime = Path(runtime_directory)
        environment_overrides = {
            "EVIDENCE_CACHE_DIR": str(
                isolated_runtime / "evidence_cache"
            ),
            "EVIDENCE_AUDIT_PATH": str(
                isolated_runtime / "evidence_audit.jsonl"
            ),
            "SUPPORT_SIGNAL_CACHE_DIR": str(
                isolated_runtime / "support_signal_cache"
            ),
        }

        for name, module in PIPELINE_STEPS:
            step_results.append(
                run_step(
                    root,
                    name,
                    module,
                    environment_overrides,
                )
            )

    manifest = build_submission_manifest(root, step_results)
    output_path = root / MANIFEST_PATH
    write_json_atomic(output_path, manifest)

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
        "INR "
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
