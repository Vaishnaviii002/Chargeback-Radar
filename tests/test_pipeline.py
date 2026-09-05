from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.pipeline import (
    OFFLINE_REPRODUCTION_ENV,
    PIPELINE_STEPS,
    REQUIRED_OUTPUTS,
    StepResult,
    build_submission_manifest,
    run_step,
    sha256_file,
    write_json_atomic,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def build_fake_outputs(root: Path) -> None:
    (root / "data").mkdir(parents=True)
    (root / "artifacts").mkdir(parents=True)
    (root / "reports").mkdir(parents=True)

    parquet_outputs = [
        path
        for path in REQUIRED_OUTPUTS
        if path.endswith(".parquet")
    ]

    for relative_path in parquet_outputs:
        (root / relative_path).write_bytes(b"parquet-test-fixture")

    write_json(
        root / "reports/metrics.json",
        {
            "model_performance": {
                "average_precision": 0.15,
                "recall": 0.60,
            },
            "disclosures": {"split": "customer-disjoint test"},
        },
    )
    write_json(root / "reports/calibration_metrics.json", {})
    write_json(root / "reports/calibration_curve.json", {})
    write_json(root / "reports/precision_recall_curve.json", {})
    write_json(
        root / "reports/policy_default.json",
        {"costs": {"estimated_net_benefit_rupees": 1000.0}},
    )
    write_json(root / "reports/policy_frontier.json", {})
    write_json(
        root / "reports/effectiveness_sensitivity.json",
        {},
    )
    write_json(
        root / "reports/ablation.json",
        {
            "baseline": {"metrics": {}},
            "enhanced": {"metrics": {}},
            "delta_enhanced_minus_baseline": {},
        },
    )
    write_json(
        root / "reports/evidence_guardrail_eval.json",
        {
            "schema_version": "1.0",
            "summary": {"passed_scenarios": 1},
        },
    )
    write_json(
        root / "reports/model_card.json",
        {"model_card_version": "model-card-v1"},
    )
    (root / "MODEL_CARD.md").write_text(
        "# Model card\n",
        encoding="utf-8",
    )
    write_json(
        root / "artifacts/feature_schema.json",
        {
            "target": "chargeback_within_120d",
            "model_features": ["amount_paise"],
            "excluded_rule_fields": ["is_duplicate_payment"],
        },
    )
    write_json(
        root / "artifacts/model_metadata.json",
        {
            "model_version": "0.1.0",
            "customer_disjoint": True,
        },
    )
    (root / "artifacts/model_bundle.joblib").write_bytes(b"model")
    (root / "artifacts/calibrator.joblib").write_bytes(b"calibrator")


def fake_read_parquet(
    path: Path,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    del columns
    filename = Path(path).name

    if filename == "payments.parquet":
        return pd.DataFrame(
            {
                "payment_id": ["pay_1", "pay_2"],
                "chargeback_within_120d": [0, 1],
            }
        )

    if filename == "disputes.parquet":
        return pd.DataFrame({"payment_id": ["pay_2"]})

    if filename == "operations.parquet":
        return pd.DataFrame(
            {
                "payment_id": ["pay_1", "pay_2"],
                "shipment_sla_breached": [False, True],
                "refund_requested_at": [
                    pd.NaT,
                    pd.Timestamp("2026-01-01"),
                ],
                "promised_refund_not_processed": [False, True],
            }
        )

    if filename == "test_explanations.parquet":
        return pd.DataFrame(
            {"explanation_version": ["shap-v1"]}
        )

    if filename == "test_model_explanations.parquet":
        return pd.DataFrame(
            {"text_explanation_version": ["plain-v1"]}
        )

    raise AssertionError(f"Unexpected parquet read: {path}")


def test_pipeline_order_preserves_evaluation_integrity() -> None:
    modules = [module for _, module in PIPELINE_STEPS]

    assert modules.index("src.outcomes") < modules.index("src.features")
    assert modules.index("src.train") < modules.index("src.calibrate")
    assert modules.index("src.calibrate") < modules.index("src.evaluate")
    assert modules.index("src.evaluate") < modules.index("src.decide")
    assert modules.index("src.explain") < modules.index(
        "src.model_explanation_service"
    )
    assert modules.index("src.support_events") < modules.index(
        "src.support_signal_report"
    )
    assert modules.index("src.support_signal_report") < modules.index(
        "src.ablation"
    )
    assert modules.index("src.ablation") < modules.index("src.model_card")


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "artifact.txt"
    path.write_bytes(b"chargeback-radar")

    assert sha256_file(path) == hashlib.sha256(
        b"chargeback-radar"
    ).hexdigest()


def test_pipeline_steps_force_offline_safe_configuration(
    tmp_path: Path,
) -> None:
    with patch("src.pipeline.subprocess.run") as run:
        run_step(tmp_path, "Safe stage", "src.safe_stage")

    environment = run.call_args.kwargs["env"]
    assert all(
        environment[name] == value
        for name, value in OFFLINE_REPRODUCTION_ENV.items()
    )


def test_pipeline_step_applies_isolated_runtime_paths(
    tmp_path: Path,
) -> None:
    isolated_cache = tmp_path / "isolated-cache"

    with patch("src.pipeline.subprocess.run") as run:
        run_step(
            tmp_path,
            "Safe stage",
            "src.safe_stage",
            {"SUPPORT_SIGNAL_CACHE_DIR": str(isolated_cache)},
        )

    assert (
        run.call_args.kwargs["env"]["SUPPORT_SIGNAL_CACHE_DIR"]
        == str(isolated_cache)
    )


def test_atomic_json_write_replaces_complete_document(
    tmp_path: Path,
) -> None:
    output = tmp_path / "manifest.json"
    output.write_text("old", encoding="utf-8")

    write_json_atomic(output, {"status": "complete"})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "status": "complete"
    }
    assert list(tmp_path.glob("*.tmp")) == []


def test_manifest_consolidates_submission_evidence(
    tmp_path: Path,
) -> None:
    build_fake_outputs(tmp_path)
    steps = [
        StepResult("Test step", "src.test", "completed", 0.1)
    ]

    with patch(
        "src.pipeline.pd.read_parquet",
        side_effect=fake_read_parquet,
    ):
        manifest = build_submission_manifest(tmp_path, steps)

    assert manifest["dataset"]["payments"] == 2
    assert manifest["dataset"]["chargebacks"] == 1
    assert manifest["dataset"]["chargeback_rate"] == 0.5
    assert manifest["evaluation_integrity"][
        "customer_disjoint"
    ] is True
    assert manifest["defense_only_controls"][
        "automatic_financial_actions"
    ] is False
    assert manifest["versions"] == {
        "model": "0.1.0",
        "dataset": None,
        "shap_explanation": "shap-v1",
        "text_explanation": "plain-v1",
        "evidence_schema": "1.0",
        "razorpay_normalization": "razorpay-normalizer-v1",
    }
    assert len(manifest["artifact_inventory"]) == len(
        REQUIRED_OUTPUTS
    )


def test_artifact_inventory_contains_real_hashes(
    tmp_path: Path,
) -> None:
    build_fake_outputs(tmp_path)

    with patch(
        "src.pipeline.pd.read_parquet",
        side_effect=fake_read_parquet,
    ):
        manifest = build_submission_manifest(tmp_path, [])

    assert all(
        len(artifact["sha256"]) == 64
        for artifact in manifest["artifact_inventory"]
    )
    assert all(
        artifact["bytes"] > 0
        for artifact in manifest["artifact_inventory"]
    )
