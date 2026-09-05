from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_TEST_ROWS = 12_173
EXPECTED_TOTAL_ROWS = 80_000
EXPECTED_MODEL_VERSION = "0.1.0"
EXPECTED_TARGET = "chargeback_within_120d"
EXPECTED_SHAP_VERSION = "shap-v1"
EXPECTED_TEXT_VERSION = "plain-v1"
EXPECTED_SIGNALS = {
    "intent_to_cancel",
    "non_receipt_complaint",
    "dissatisfaction",
}
EXPECTED_RAZORPAY_PATHS = {
    "/api/razorpay-test/status",
    "/api/razorpay-test/webhooks",
    "/api/razorpay-test/orders",
    "/api/razorpay-test/orders/{order_id}",
    (
        "/api/razorpay-test/orders/"
        "{order_id}/verify-checkout"
    ),
    (
        "/api/razorpay-test/orders/"
        "{order_id}/verify-and-score"
    ),
    "/api/razorpay-test/payments/{payment_id}",
    (
        "/api/razorpay-test/payments/"
        "{payment_id}/score"
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(left: Any, right: Any) -> bool:
    return math.isclose(
        float(left),
        float(right),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def load_json(relative_path: str) -> dict[str, Any]:
    with open(
        ROOT / relative_path,
        "r",
        encoding="utf-8",
    ) as handle:
        value = json.load(handle)

    require(
        isinstance(value, dict),
        f"{relative_path} must contain an object.",
    )
    return value


def run(command: list[str], label: str) -> None:
    print(f"RUN: {label}", flush=True)
    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )
    print(f"PASS: {label}", flush=True)


def validate_required_files() -> None:
    required = {
        "artifacts/model_bundle.joblib",
        "artifacts/calibrator.joblib",
        "artifacts/model_metadata.json",
        "artifacts/feature_schema.json",
        "reports/metrics.json",
        "reports/calibration_metrics.json",
        "reports/test_predictions.parquet",
        "reports/test_scored.parquet",
        "reports/test_explanations.parquet",
        "reports/test_model_explanations.parquet",
        "reports/support_signals.parquet",
        "reports/ablation.json",
        "reports/ablation_test_predictions.parquet",
        "reports/model_card.json",
    }
    missing = sorted(
        path
        for path in required
        if not (ROOT / path).is_file()
    )
    require(
        not missing,
        "Missing required release files: "
        + ", ".join(missing),
    )
    print("PASS: required model and report files", flush=True)


def validate_model_and_explanations() -> None:
    model_bundle = joblib.load(
        ROOT / "artifacts/model_bundle.joblib"
    )
    calibrator_bundle = joblib.load(
        ROOT / "artifacts/calibrator.joblib"
    )
    metadata = load_json(
        "artifacts/model_metadata.json"
    )
    metrics = load_json("reports/metrics.json")
    calibration = load_json(
        "reports/calibration_metrics.json"
    )

    require(
        model_bundle["model_version"]
        == EXPECTED_MODEL_VERSION
        == metadata["model_version"],
        "Model version identity changed.",
    )
    require(
        model_bundle["target"] == EXPECTED_TARGET,
        "Model target changed.",
    )
    require(
        calibrator_bundle["method"] == "isotonic"
        == calibration["selected_method"],
        "Authoritative calibration is not isotonic.",
    )
    require(
        metrics["dataset"]["test_rows"]
        == EXPECTED_TEST_ROWS,
        "Held-out row count changed.",
    )

    predictions = pd.read_parquet(
        ROOT / "reports/test_predictions.parquet"
    )
    scored = pd.read_parquet(
        ROOT / "reports/test_scored.parquet"
    )
    shap_report = pd.read_parquet(
        ROOT / "reports/test_explanations.parquet"
    )
    text_report = pd.read_parquet(
        ROOT / "reports/test_model_explanations.parquet"
    )

    reports = {
        "predictions": predictions,
        "scored": scored,
        "SHAP explanations": shap_report,
        "text explanations": text_report,
    }
    authoritative_ids = predictions[
        "payment_id"
    ].astype(str).tolist()

    for name, report in reports.items():
        require(
            len(report) == EXPECTED_TEST_ROWS,
            f"{name} row count changed.",
        )
        ids = report["payment_id"].astype(str)
        require(
            ids.is_unique,
            f"{name} payment IDs are not unique.",
        )
        require(
            ids.tolist() == authoritative_ids,
            f"{name} payment IDs are not exactly aligned.",
        )

    require(
        set(shap_report["model_version"].astype(str))
        == {EXPECTED_MODEL_VERSION},
        "SHAP model version changed.",
    )
    require(
        set(shap_report["explanation_version"].astype(str))
        == {EXPECTED_SHAP_VERSION},
        "SHAP explanation version changed.",
    )
    require(
        set(shap_report["target"].astype(str))
        == {EXPECTED_TARGET},
        "SHAP target metadata changed.",
    )
    require(
        float(
            shap_report[
                "shap_reconstruction_error"
            ].abs().max()
        )
        <= 1e-8,
        "SHAP additivity tolerance was exceeded.",
    )

    forbidden_columns = {
        "customer_id",
        "customer_name",
        "customer_email",
        "customer_phone",
        "dispute_id",
        "dispute_status",
        "dispute_reason",
        "chargeback_within_120d",
        "true_fraud",
        "final_outcome",
    }
    require(
        forbidden_columns.isdisjoint(
            shap_report.columns
        ),
        "Forbidden identity or outcome columns reached SHAP.",
    )
    require(
        forbidden_columns.isdisjoint(
            text_report.columns
        ),
        "Forbidden identity or outcome columns reached text explanations.",
    )

    factor_columns = (
        "top_positive_factors_json",
        "top_negative_factors_json",
        "all_feature_contributions_json",
    )
    forbidden_factor_names = forbidden_columns | {
        "payment_id",
        "order_id",
        "target",
        "reason_code",
        "refund_status",
    }

    for row in shap_report.itertuples(index=False):
        parsed: dict[str, list[dict[str, Any]]] = {}

        for column in factor_columns:
            factors = json.loads(getattr(row, column))
            require(
                isinstance(factors, list),
                f"{column} must contain a list.",
            )
            parsed[column] = factors

            for factor in factors:
                require(
                    {"feature", "value", "shap_value", "direction"}
                    <= set(factor),
                    "Explanation factor fields are incomplete.",
                )
                require(
                    factor["feature"]
                    not in forbidden_factor_names,
                    "Forbidden feature reached an explanation.",
                )

        positive = parsed["top_positive_factors_json"]
        negative = parsed["top_negative_factors_json"]
        positive_values = [
            float(factor["shap_value"])
            for factor in positive
        ]
        negative_values = [
            float(factor["shap_value"])
            for factor in negative
        ]
        require(
            all(value > 0 for value in positive_values)
            and positive_values
            == sorted(positive_values, reverse=True),
            "Risk-increasing SHAP ordering/direction changed.",
        )
        require(
            all(value < 0 for value in negative_values)
            and negative_values == sorted(negative_values),
            "Risk-decreasing SHAP ordering/direction changed.",
        )
        require(
            all(
                factor["direction"] == "increases_risk"
                for factor in positive
            )
            and all(
                factor["direction"] == "decreases_risk"
                for factor in negative
            ),
            "SHAP direction labels changed.",
        )

    require(
        set(text_report["model_version"].astype(str))
        == {EXPECTED_MODEL_VERSION},
        "Delivered explanation model version changed.",
    )
    require(
        set(
            text_report[
                "shap_explanation_version"
            ].astype(str)
        )
        == {EXPECTED_SHAP_VERSION},
        "Delivered SHAP version changed.",
    )
    require(
        set(
            text_report[
                "text_explanation_version"
            ].astype(str)
        )
        == {EXPECTED_TEXT_VERSION},
        "Deterministic explanation version changed.",
    )
    require(
        set(text_report["label"].astype(str))
        == {"Model explanation — not evidence"},
        "Required model-explanation label changed.",
    )
    combined_text = " ".join(
        text_report["explanation"].astype(str)
    ).lower()
    require(
        "donot establish causation" not in combined_text
        and "do not establish causation" in combined_text,
        "Model-explanation disclaimer text is malformed.",
    )
    print(
        "PASS: model identity, held-out alignment, SHAP additivity, "
        "factor governance and explanation delivery",
        flush=True,
    )


def validate_support_ablation_and_model_card() -> None:
    signals = pd.read_parquet(
        ROOT / "reports/support_signals.parquet"
    )
    require(
        len(signals) == EXPECTED_TOTAL_ROWS,
        "Support-signal row count changed.",
    )
    require(
        EXPECTED_SIGNALS <= set(signals.columns),
        "Required support signals are missing.",
    )
    forbidden_support = {
        "customer_id",
        "customer_name",
        "customer_email",
        "customer_phone",
        "message_text",
        "support_event_id",
        "chargeback_within_120d",
        "true_fraud",
        "dispute_id",
        "dispute_status",
        "reason_code",
        "chargeback_outcome",
    }
    require(
        forbidden_support.isdisjoint(signals.columns),
        "Raw text, identity or outcomes reached support signals.",
    )

    for signal in EXPECTED_SIGNALS:
        require(
            set(signals[signal].dropna().astype(int))
            <= {0, 1},
            f"{signal} is not binary.",
        )

    require(
        int(signals["observable_event_count"].sum())
        == 8_270,
        "Observable support-event count changed.",
    )
    require(
        int(signals["excluded_future_event_count"].sum())
        == 5_297,
        "Future-event exclusion count changed.",
    )

    metrics = load_json("reports/metrics.json")
    calibration = load_json(
        "reports/calibration_metrics.json"
    )
    ablation = load_json("reports/ablation.json")
    model_card = load_json("reports/model_card.json")
    baseline = ablation["baseline"]["metrics"]
    canonical = metrics["model_performance"]

    for name in (
        "average_precision",
        "precision",
        "recall",
        "f1",
        "expected_calibration_error",
        "precision_at_1_percent",
        "precision_at_5_percent",
    ):
        require(
            close(baseline[name], canonical[name]),
            f"Ablation baseline differs for {name}.",
        )

    require(
        close(
            baseline["brier_score"],
            calibration["test_calibrated_brier"],
        ),
        "Ablation baseline Brier score changed.",
    )
    require(
        close(
            ablation[
                "canonical_baseline_comparison"
            ]["maximum_absolute_difference"],
            0,
        ),
        "Ablation canonical comparison is not exact.",
    )

    card_evaluation = model_card[
        "canonical_evaluation"
    ]
    for name in (
        "average_precision",
        "precision",
        "recall",
        "f1",
        "expected_calibration_error",
        "precision_at_1_percent",
        "precision_at_5_percent",
    ):
        require(
            close(card_evaluation[name], canonical[name]),
            f"Model card differs for {name}.",
        )

    require(
        model_card["target"]["name"]
        == EXPECTED_TARGET
        and model_card["target"]["window_days"]
        == 120,
        "Model-card target changed.",
    )
    require(
        model_card["data"]["synthetic"] is True
        and model_card["data"]["customer_disjoint"]
        is True,
        "Model-card data disclosures changed.",
    )
    human = model_card["human_control"]
    require(
        human["human_approval_required"] is True
        and human["automatic_refund"] is False
        and human["automatic_customer_message"] is False
        and human["automatic_dispute_submission"] is False,
        "Human-control boundary changed.",
    )
    explainability = model_card["explainability"]
    require(
        explainability["causal"] is False
        and explainability["dispute_evidence"] is False,
        "Model-explanation governance changed.",
    )
    print(
        "PASS: support signals, future exclusion, ablation baseline "
        "and model-card consistency",
        flush=True,
    )


def validate_openapi() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        from src.api import app

        schema = app.openapi()

    paths = set(schema["paths"])
    require(
        EXPECTED_RAZORPAY_PATHS <= paths,
        "One or more intended Razorpay routes are missing.",
    )
    operation_ids = [
        operation["operationId"]
        for path_item in schema["paths"].values()
        for operation in path_item.values()
        if isinstance(operation, dict)
        and "operationId" in operation
    ]
    require(
        len(operation_ids) == len(set(operation_ids)),
        "OpenAPI contains duplicate operation IDs.",
    )
    print(
        f"PASS: OpenAPI ({len(paths)} paths, "
        f"{len(operation_ids)} unique operation IDs)",
        flush=True,
    )


def git_output(arguments: list[str]) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout


def validate_repository_hygiene() -> None:
    tracked = {
        path
        for path in git_output(
            ["ls-files"]
        ).splitlines()
        if path
    }
    reviewable = tracked | {
        path
        for path in git_output(
            ["ls-files", "--others", "--exclude-standard"]
        ).splitlines()
        if path
    }
    forbidden_tracked = sorted(
        path
        for path in tracked
        if path in {".env", ".env.local", "frontend/.env"}
        or path.startswith("runtime/")
        or path.startswith("frontend/dist/")
        or "/__pycache__/" in f"/{path}"
        or Path(path).suffix.lower()
        in {".db", ".log", ".pyc", ".sqlite", ".sqlite3"}
    )
    require(
        not forbidden_tracked,
        "Runtime or secret files are tracked: "
        + ", ".join(forbidden_tracked),
    )

    for candidate in (
        ".env",
        ".env.local",
        "frontend/.env",
        "runtime/razorpay.sqlite3",
        "runtime/evidence_audit.jsonl",
        "frontend/dist/index.html",
    ):
        ignored = subprocess.run(
            ["git", "check-ignore", "--quiet", candidate],
            cwd=ROOT,
            check=False,
        ).returncode == 0
        require(
            ignored,
            f"{candidate} is not ignored.",
        )

    secret_patterns = {
        "OpenAI key": re.compile(
            r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"
        ),
        "Razorpay key": re.compile(
            r"rzp_(?:test|live)_[A-Za-z0-9]{14,}"
        ),
        "temporary tunnel URL": re.compile(
            r"https://[^\s/]+\.(?:trycloudflare\.com|ngrok\.app)",
            re.IGNORECASE,
        ),
    }
    text_suffixes = {
        ".css",
        ".dockerignore",
        ".env",
        ".example",
        ".html",
        ".json",
        ".md",
        ".ps1",
        ".py",
        ".ts",
        ".tsx",
        ".txt",
        ".yaml",
        ".yml",
    }

    for relative_path in sorted(reviewable):
        path = ROOT / relative_path

        if (
            path.name not in {"Dockerfile", ".gitignore"}
            and path.suffix.lower() not in text_suffixes
        ):
            continue

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for label, pattern in secret_patterns.items():
            require(
                pattern.search(content) is None,
                f"Possible {label} in {relative_path}.",
            )

    env_example = (
        ROOT / ".env.example"
    ).read_text(encoding="utf-8")
    for secret_name in (
        "OPENAI_API_KEY",
        "EVIDENCE_AUDIT_HMAC_KEY",
        "RAZORPAY_KEY_ID",
        "RAZORPAY_KEY_SECRET",
        "RAZORPAY_WEBHOOK_SECRET",
    ):
        match = re.search(
            rf"(?m)^{secret_name}=(.*)$",
            env_example,
        )
        require(
            match is not None
            and not match.group(1).strip(),
            f"{secret_name} must be blank in .env.example.",
        )

    run(
        ["git", "diff", "--check"],
        "unstaged Git whitespace check",
    )
    run(
        ["git", "diff", "--cached", "--check"],
        "staged Git whitespace check",
    )
    print(
        "PASS: tracked-file, ignore, credential and tunnel hygiene",
        flush=True,
    )


def main() -> None:
    print("Chargeback Radar release verification", flush=True)
    validate_required_files()
    validate_model_and_explanations()
    validate_support_ablation_and_model_card()
    validate_openapi()
    validate_repository_hygiene()
    run(
        [sys.executable, "-m", "pytest", "-q"],
        "backend test suite",
    )
    npm = shutil.which("npm")
    require(
        npm is not None,
        "npm is required for the frontend build.",
    )
    run(
        [npm, "--prefix", "frontend", "run", "lint"],
        "frontend lint",
    )
    run(
        [npm, "--prefix", "frontend", "run", "build"],
        "frontend production build",
    )
    print("RELEASE VERIFICATION: PASS", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (
        AssertionError,
        FileNotFoundError,
        KeyError,
        OSError,
        subprocess.CalledProcessError,
        ValueError,
    ) as error:
        print(
            f"RELEASE VERIFICATION: FAIL — {error}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from error
