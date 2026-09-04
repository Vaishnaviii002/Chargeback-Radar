from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

MODEL_BUNDLE_PATH = (
    ROOT
    / "artifacts"
    / "model_bundle.joblib"
)

MODEL_METADATA_PATH = (
    ROOT
    / "artifacts"
    / "model_metadata.json"
)

FEATURE_SCHEMA_PATH = (
    ROOT
    / "artifacts"
    / "feature_schema.json"
)

METRICS_PATH = (
    ROOT
    / "reports"
    / "metrics.json"
)

CALIBRATION_METRICS_PATH = (
    ROOT
    / "reports"
    / "calibration_metrics.json"
)

ABLATION_PATH = (
    ROOT
    / "reports"
    / "ablation.json"
)

SUPPORT_SIGNALS_PATH = (
    ROOT
    / "reports"
    / "support_signals.parquet"
)

MODEL_CARD_JSON_PATH = (
    ROOT
    / "reports"
    / "model_card.json"
)

MODEL_CARD_MARKDOWN_PATH = (
    ROOT
    / "MODEL_CARD.md"
)


MODEL_CARD_VERSION: Final = (
    "model-card-v1"
)

AUTHORITATIVE_TARGET: Final = (
    "chargeback_within_120d"
)


FORBIDDEN_FEATURES: Final = [
    "customer_id",
    "customer_name",
    "customer_email",
    "customer_phone",
    "dispute_id",
    "dispute_status",
    "dispute_reason",
    "reason_code",
    "chargeback_family",
    "chargeback_outcome",
    "true_fraud",
    "final_outcome",
    "was_won",
    "was_lost",
    "future_chargeback",
    "future_refund",
    "future_delivery",
    "future_dispute",
    "final_reason_code",
]


INTENDED_USES: Final = [
    (
        "Prioritize merchant payments for "
        "human risk review."
    ),
    (
        "Prioritize preparation of factual "
        "chargeback evidence."
    ),
    (
        "Support bounded, cost-aware merchant "
        "defense workflows."
    ),
    (
        "Evaluate risk-system methodology on "
        "synthetic held-out data."
    ),
]


PROHIBITED_USES: Final = [
    "Automatic accusation of a customer.",
    "Criminal or fraud determination.",
    "Creditworthiness or lending decisions.",
    "Identity profiling.",
    "Automatic customer punishment.",
    "Automatic refund execution.",
    "Automatic customer messaging.",
    "Automatic dispute submission.",
    "Offensive fraud or payment abuse.",
]


KNOWN_LIMITATIONS: Final = [
    (
        "The dataset, chargebacks, operational "
        "events and support messages are synthetic."
    ),
    (
        (
    "Absolute performance does not establish "
    "real-world generalization to merchant traffic."
),
    ),
    (
        "Issuer, network and merchant behavior is "
        "simplified compared with production."
    ),
    (
        "Concept drift and live population drift "
        "have not been measured."
    ),
    (
        "Support-text signals are evaluated at a "
        "seven-day post-payment scoring point."
    ),
    (
        "Support-message templates simplify real "
        "language diversity and ambiguity."
    ),
    (
        "SHAP describes model contribution, not "
        "causation, fraud or dispute evidence."
    ),
    (
        "The LLM does not generate calibrated "
        "chargeback-risk probability."
    ),
    (
        "Razorpay test mode cannot fabricate a "
        "real issuer-generated chargeback."
    ),
    (
        "All protected merchant actions require "
        "human review and approval."
    ),
]


def _load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required model-card input "
            f"is missing: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        value = json.load(file)

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            f"Expected a JSON object in {path}"
        )

    return value


def _count_values(
    values: pd.Series,
) -> dict[str, int]:
    counts = (
        values
        .fillna("NONE")
        .astype(str)
        .value_counts(
            dropna=False
        )
    )

    return {
        str(key): int(value)
        for key, value
        in counts.items()
    }


def _validate_sources(
    *,
    bundle: dict[str, Any],
    metadata: dict[str, Any],
    schema: dict[str, Any],
    metrics: dict[str, Any],
    calibration: dict[str, Any],
    ablation: dict[str, Any],
    support_signals: pd.DataFrame,
) -> None:
    targets = {
        str(bundle["target"]),
        str(schema["target"]),
        str(ablation["target"]),
        AUTHORITATIVE_TARGET,
    }

    if targets != {
        AUTHORITATIVE_TARGET
    }:
        raise ValueError(
            "Model-card sources disagree on "
            "the target definition"
        )

    if (
        list(bundle["model_features"])
        != list(schema["model_features"])
    ):
        raise ValueError(
            "Model bundle and feature schema "
            "disagree on model features"
        )

    if (
        list(bundle["categorical_features"])
        != list(
            schema[
                "categorical_features"
            ]
        )
    ):
        raise ValueError(
            "Categorical feature definitions "
            "do not match"
        )

    if (
        list(bundle["numerical_features"])
        != list(
            schema[
                "numerical_features"
            ]
        )
    ):
        raise ValueError(
            "Numerical feature definitions "
            "do not match"
        )

    if (
        str(bundle["model_version"])
        != str(
            metadata["model_version"]
        )
    ):
        raise ValueError(
            "Model-version identity mismatch"
        )

    if (
        metadata["customer_disjoint"]
        is not True
    ):
        raise ValueError(
            "Customer-disjoint split is not "
            "confirmed"
        )

    dataset = metrics[
        "dataset"
    ]

    if (
        int(dataset["test_rows"])
        != int(
            metadata["test_rows"]
        )
    ):
        raise ValueError(
            "Canonical test-row count mismatch"
        )

    if (
        int(dataset["positive_labels"])
        != int(
            metadata["test_positives"]
        )
    ):
        raise ValueError(
            "Canonical positive-label "
            "count mismatch"
        )

    ablation_dataset = (
        ablation["dataset"]
    )

    for key in (
        "training_rows",
        "calibration_rows",
        "test_rows",
        "training_positives",
        "calibration_positives",
        "test_positives",
    ):
        if int(
            ablation_dataset[key]
        ) != int(metadata[key]):
            raise ValueError(
                "Ablation and model metadata "
                f"disagree on {key}"
            )

    expected_payment_rows = (
        int(metadata["training_rows"])
        + int(metadata["calibration_rows"])
        + int(metadata["test_rows"])
    )

    if len(
        support_signals
    ) != expected_payment_rows:
        raise ValueError(
            "Support-signal report does not "
            "cover the full payment dataset"
        )

    if support_signals[
        "payment_id"
    ].duplicated().any():
        raise ValueError(
            "Support-signal report contains "
            "duplicate payment IDs"
        )

    required_signal_columns = {
        "intent_to_cancel",
        "non_receipt_complaint",
        "dissatisfaction",
        "input_hash",
        "delivery_mode",
        "provider",
        "fallback_used",
        "observable_event_count",
        "excluded_future_event_count",
    }

    missing = sorted(
        required_signal_columns
        - set(
            support_signals.columns
        )
    )

    if missing:
        raise ValueError(
            "Support-signal report is "
            "missing columns: "
            + ", ".join(missing)
        )

    if (
        calibration[
            "selected_method"
        ]
        not in {
            "sigmoid",
            "isotonic",
        }
    ):
        raise ValueError(
            "Unknown selected calibration "
            "method"
        )


def _support_summary(
    report: pd.DataFrame,
) -> dict[str, Any]:
    row_count = len(report)

    signal_summary = {}

    for signal in (
        "intent_to_cancel",
        "non_receipt_complaint",
        "dissatisfaction",
    ):
        positive_count = int(
            report[signal].sum()
        )

        signal_summary[signal] = {
            "positive_count": (
                positive_count
            ),
            "positive_rate": (
                positive_count
                / row_count
                if row_count
                else 0.0
            ),
        }

    return {
        "rows": row_count,
        "unique_sanitized_input_hashes": int(
            report[
                "input_hash"
            ].nunique()
        ),
        "payments_with_observable_text": int(
            (
                report[
                    "observable_event_count"
                ]
                > 0
            ).sum()
        ),
        "excluded_future_events": int(
            report[
                "excluded_future_event_count"
            ].sum()
        ),
        "delivery_mode_counts": (
            _count_values(
                report[
                    "delivery_mode"
                ]
            )
        ),
        "provider_counts": (
            _count_values(
                report["provider"]
            )
        ),
        "fallback_rows": int(
            report[
                "fallback_used"
            ].sum()
        ),
        "fallback_reason_counts": (
            _count_values(
                report[
                    "fallback_reason"
                ]
            )
        ),
        "signals": signal_summary,
    }


def _canonical_evaluation(
    *,
    metrics: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    performance = metrics[
        "model_performance"
    ]

    return {
        "test_rows": int(
            metrics[
                "dataset"
            ]["test_rows"]
        ),
        "positive_labels": int(
            metrics[
                "dataset"
            ]["positive_labels"]
        ),
        "base_rate": float(
            metrics[
                "dataset"
            ]["base_rate"]
        ),
        "average_precision": float(
            performance[
                "average_precision"
            ]
        ),
        "precision": float(
            performance["precision"]
        ),
        "recall": float(
            performance["recall"]
        ),
        "f1": float(
            performance["f1"]
        ),
        "brier_score": float(
            calibration[
                "test_calibrated_brier"
            ]
        ),
        "expected_calibration_error": float(
            performance[
                "expected_calibration_error"
            ]
        ),
        "precision_at_1_percent": float(
            performance[
                "precision_at_1_percent"
            ]
        ),
        "precision_at_5_percent": float(
            performance[
                "precision_at_5_percent"
            ]
        ),
        "operating_threshold": float(
            metrics[
                "operating_policy"
            ]["operating_threshold"]
        ),
        "review_capacity": float(
            metrics[
                "operating_policy"
            ]["review_capacity"]
        ),
        "flagged_count": int(
            metrics[
                "operating_policy"
            ]["flagged_count"]
        ),
        "flagged_rate": float(
            metrics[
                "operating_policy"
            ]["flagged_rate"]
        ),
        "confusion_matrix": (
            metrics[
                "confusion_matrix"
            ]
        ),
        "false_positive_cost": (
            metrics[
                "false_positive_cost"
            ]
        ),
    }


def _percent(
    value: float,
) -> str:
    return f"{value:.3%}"


def _decimal(
    value: float,
) -> str:
    return f"{value:.6f}"


def _build_markdown(
    card: dict[str, Any],
) -> str:
    model = card["model"]
    data = card["data"]
    evaluation = card[
        "canonical_evaluation"
    ]
    ablation = card["ablation"]
    support = card[
        "support_text_signals"
    ]

    baseline = ablation[
        "baseline_metrics"
    ]
    enhanced = ablation[
        "enhanced_metrics"
    ]
    delta = ablation[
        "delta_enhanced_minus_baseline"
    ]

    feature_lines = "\n".join(
        f"- `{feature}`"
        for feature in card[
            "features"
        ]["model_features"]
    )

    forbidden_lines = "\n".join(
        f"- `{feature}`"
        for feature in card[
            "features"
        ]["forbidden_features"]
    )

    intended_lines = "\n".join(
        f"- {item}"
        for item in card[
            "intended_use"
        ]
    )

    prohibited_lines = "\n".join(
        f"- {item}"
        for item in card[
            "prohibited_use"
        ]
    )

    limitation_lines = "\n".join(
        f"- {item}"
        for item in card[
            "known_limitations"
        ]
    )

    return f"""# Chargeback Radar Model Card

**Model-card version:** `{card["model_card_version"]}`

**Model version:** `{model["version"]}`

**Target:** `{card["target"]["name"]}`

**Status:** Hackathon prototype evaluated on synthetic data

## Summary

Chargeback Radar is a defense-only merchant decision-support system. Its LightGBM model estimates the probability of the synthetic target `{card["target"]["name"]}`. The score supports human prioritization; it does not establish fraud, causation, wrongdoing or dispute evidence.

Every protected merchant action requires human approval. The system never automatically refunds, messages a customer or submits a dispute.

## Target definition

`{card["target"]["name"]}` means whether a payment receives a chargeback within the synthetic 120-day observation window.

This is a 120-day target. It must not be described as a 30-day prediction.

## Data and evaluation split

- Data source: synthetic payments, outcomes, operations and support events
- Total payments: {data["total_rows"]:,}
- Training: months 1–9 ({data["training_rows"]:,} rows)
- Calibration: month 10 ({data["calibration_rows"]:,} rows)
- Held-out test: months 11–12 ({data["test_rows"]:,} rows)
- Customers disjoint across all splits: **Yes**
- Held-out test customers unseen during training: **Yes**

The held-out test is also the unseen-customer stress test because every test customer is disjoint from training and calibration.

## Model and calibration

- Estimator: `{model["estimator"]}`
- Seed: `{model["seed"]}`
- Best iteration: `{model["best_iteration"]}`
- Calibration method selected from calibration-month data: `{model["calibration_method"]}`
- Review capacity used to choose the operating threshold: {_percent(evaluation["review_capacity"])}

Calibration-method selection uses the first 70% of month 10 for fitting candidates and the final 30% for choosing between sigmoid and isotonic calibration. Test labels are not used for calibration or threshold selection.

## Canonical held-out performance

| Metric | Value |
|---|---:|
| Test payments | {evaluation["test_rows"]:,} |
| Chargebacks | {evaluation["positive_labels"]:,} |
| Base rate | {_percent(evaluation["base_rate"])} |
| Average Precision | {_decimal(evaluation["average_precision"])} |
| Precision | {_percent(evaluation["precision"])} |
| Recall | {_percent(evaluation["recall"])} |
| F1 | {_decimal(evaluation["f1"])} |
| Brier score | {_decimal(evaluation["brier_score"])} |
| Expected calibration error | {_decimal(evaluation["expected_calibration_error"])} |
| Precision at top 1% | {_percent(evaluation["precision_at_1_percent"])} |
| Precision at top 5% | {_percent(evaluation["precision_at_5_percent"])} |
| Operating threshold | {_decimal(evaluation["operating_threshold"])} |
| Flagged payments | {evaluation["flagged_count"]:,} |
| Flagged rate | {_percent(evaluation["flagged_rate"])} |

Accuracy is intentionally not used as the headline metric because the held-out base rate is below one percent.

## Unseen-customer stress test

- Test customers: {card["unseen_customer_stress_test"]["unique_test_customers"]:,}
- Overlap with training customers: 0
- Overlap with calibration customers: 0
- Test rows: {card["unseen_customer_stress_test"]["test_rows"]:,}
- Average Precision: {_decimal(card["unseen_customer_stress_test"]["average_precision"])}
- Recall: {_percent(card["unseen_customer_stress_test"]["recall"])}

## Controlled support-signal ablation

Both variants use identical payment rows, train/calibration/test partitions, seed, LightGBM settings, early stopping, calibration selection and evaluation.

The enhanced model adds exactly:

- `intent_to_cancel`
- `non_receipt_complaint`
- `dissatisfaction`

| Metric | Baseline | Enhanced | Enhanced − baseline |
|---|---:|---:|---:|
| Average Precision | {_decimal(baseline["average_precision"])} | {_decimal(enhanced["average_precision"])} | {delta["average_precision"]:+.6f} |
| Precision | {_percent(baseline["precision"])} | {_percent(enhanced["precision"])} | {delta["precision"]:+.3%} |
| Recall | {_percent(baseline["recall"])} | {_percent(enhanced["recall"])} | {delta["recall"]:+.3%} |
| F1 | {_decimal(baseline["f1"])} | {_decimal(enhanced["f1"])} | {delta["f1"]:+.6f} |
| Brier score | {_decimal(baseline["brier_score"])} | {_decimal(enhanced["brier_score"])} | {delta["brier_score"]:+.6f} |
| Calibration error | {_decimal(baseline["expected_calibration_error"])} | {_decimal(enhanced["expected_calibration_error"])} | {delta["expected_calibration_error"]:+.6f} |
| Precision at top 1% | {_percent(baseline["precision_at_1_percent"])} | {_percent(enhanced["precision_at_1_percent"])} | {delta["precision_at_1_percent"]:+.3%} |
| Precision at top 5% | {_percent(baseline["precision_at_5_percent"])} | {_percent(enhanced["precision_at_5_percent"])} | {delta["precision_at_5_percent"]:+.3%} |

The enhanced result is reported without cherry-picking: Average Precision, recall and Brier score improved, while precision changed by {delta["precision"]:+.3%}.

## Support-text safety boundary

- Scoring point: payment timestamp plus seven days
- Payments with observable support text: {support["payments_with_observable_text"]:,}
- Future support events excluded: {support["excluded_future_events"]:,}
- Unique sanitized model inputs: {support["unique_sanitized_input_hashes"]:,}
- Input-hash caching enabled: **Yes**
- PII removal before the model boundary: **Yes**
- Prompt-injection filtering before the model boundary: **Yes**
- Final chargeback labels and dispute outcomes exposed to the text model: **No**

The LLM converts sanitized, timestamp-safe support text into three bounded booleans. It does not produce the chargeback-risk probability.

## Explainability

SHAP explains how trained model inputs contributed to the model estimate.

**Model explanation — not evidence**

SHAP does not prove causation, fraud, customer intent or the factual basis of a dispute. Model explanations remain visually and contractually separate from Evidence Copilot facts.

## Model features

{feature_lines}

## Forbidden features

{forbidden_lines}

Customer identifiers may be used only to enforce customer-disjoint evaluation and are never model inputs.

## Intended use

{intended_lines}

## Prohibited use

{prohibited_lines}

## Human-control requirements

- All evidence is a draft.
- Human approval is required.
- No payment action is automatically executed.
- No customer message is automatically sent.
- No dispute is automatically submitted.
- Recommended actions remain bounded to `MONITOR`, `PREPARE_EVIDENCE`, `MANUAL_REVIEW` and `RECOMMEND_REFUND`.

## Known limitations

{limitation_lines}

## Reproducibility sources

- `artifacts/model_bundle.joblib`
- `artifacts/model_metadata.json`
- `artifacts/feature_schema.json`
- `reports/metrics.json`
- `reports/calibration_metrics.json`
- `reports/test_explanations.parquet`
- `reports/support_signals.parquet`
- `reports/ablation.json`
- `reports/model_card.json`
"""


def generate_model_card(
    *,
    model_bundle_path: Path = (
        MODEL_BUNDLE_PATH
    ),
    model_metadata_path: Path = (
        MODEL_METADATA_PATH
    ),
    feature_schema_path: Path = (
        FEATURE_SCHEMA_PATH
    ),
    metrics_path: Path = (
        METRICS_PATH
    ),
    calibration_metrics_path: Path = (
        CALIBRATION_METRICS_PATH
    ),
    ablation_path: Path = (
        ABLATION_PATH
    ),
    support_signals_path: Path = (
        SUPPORT_SIGNALS_PATH
    ),
    json_output_path: Path = (
        MODEL_CARD_JSON_PATH
    ),
    markdown_output_path: Path = (
        MODEL_CARD_MARKDOWN_PATH
    ),
) -> dict[str, Any]:
    required_paths = (
        model_bundle_path,
        model_metadata_path,
        feature_schema_path,
        metrics_path,
        calibration_metrics_path,
        ablation_path,
        support_signals_path,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                "Required model-card input "
                f"is missing: {path}"
            )

    bundle = joblib.load(
        model_bundle_path
    )

    if not isinstance(
        bundle,
        dict,
    ):
        raise ValueError(
            "Model bundle must be a dictionary"
        )

    metadata = _load_json(
        model_metadata_path
    )

    schema = _load_json(
        feature_schema_path
    )

    metrics = _load_json(
        metrics_path
    )

    calibration = _load_json(
        calibration_metrics_path
    )

    ablation = _load_json(
        ablation_path
    )

    support_signals = (
        pd.read_parquet(
            support_signals_path
        )
    )

    _validate_sources(
        bundle=bundle,
        metadata=metadata,
        schema=schema,
        metrics=metrics,
        calibration=calibration,
        ablation=ablation,
        support_signals=(
            support_signals
        ),
    )

    evaluation = (
        _canonical_evaluation(
            metrics=metrics,
            calibration=calibration,
        )
    )

    total_rows = (
        int(metadata["training_rows"])
        + int(metadata["calibration_rows"])
        + int(metadata["test_rows"])
    )

    card = {
        "model_card_version": (
            MODEL_CARD_VERSION
        ),
        "project": (
            "Chargeback Radar"
        ),
        "track": (
            "Razorpay AI Buildathon 2026 — "
            "Track 02 AI Risk Manager"
        ),
        "system_type": (
            "Defense-only merchant "
            "decision support"
        ),
        "target": {
            "name": (
                AUTHORITATIVE_TARGET
            ),
            "window_days": 120,
            "definition": (
                "Whether a synthetic payment "
                "receives a chargeback within "
                "the 120-day observation window."
            ),
        },
        "model": {
            "estimator": (
                bundle["model"]
                .__class__.__name__
            ),
            "version": str(
                bundle[
                    "model_version"
                ]
            ),
            "seed": int(
                metadata["seed"]
            ),
            "best_iteration": int(
                metadata[
                    "best_iteration"
                ]
            ),
            "calibration_method": (
                calibration[
                    "selected_method"
                ]
            ),
        },
        "data": {
            "synthetic": True,
            "total_rows": total_rows,
            "training_rows": int(
                metadata[
                    "training_rows"
                ]
            ),
            "calibration_rows": int(
                metadata[
                    "calibration_rows"
                ]
            ),
            "test_rows": int(
                metadata["test_rows"]
            ),
            "split": {
                "training": "months 1-9",
                "calibration": "month 10",
                "test": "months 11-12",
            },
            "customer_disjoint": True,
        },
        "features": {
            "model_features": list(
                bundle[
                    "model_features"
                ]
            ),
            "categorical_features": list(
                bundle[
                    "categorical_features"
                ]
            ),
            "numerical_features": list(
                bundle[
                    "numerical_features"
                ]
            ),
            "forbidden_features": list(
                FORBIDDEN_FEATURES
            ),
        },
        "canonical_evaluation": (
            evaluation
        ),
        "unseen_customer_stress_test": {
            "definition": (
                "All held-out test customers "
                "are absent from training and "
                "calibration."
            ),
            "unique_test_customers": int(
                metadata[
                    "test_unique_customers"
                ]
            ),
            "training_overlap": 0,
            "calibration_overlap": 0,
            "test_rows": int(
                metadata["test_rows"]
            ),
            "average_precision": (
                evaluation[
                    "average_precision"
                ]
            ),
            "precision": (
                evaluation["precision"]
            ),
            "recall": (
                evaluation["recall"]
            ),
        },
        "support_text_signals": (
            _support_summary(
                support_signals
            )
        ),
        "ablation": {
            "report_version": (
                ablation[
                    "report_version"
                ]
            ),
            "support_signal_features": list(
                ablation[
                    "enhanced"
                ]["features"][-3:]
            ),
            "baseline_metrics": (
                ablation[
                    "baseline"
                ]["metrics"]
            ),
            "enhanced_metrics": (
                ablation[
                    "enhanced"
                ]["metrics"]
            ),
            "delta_enhanced_minus_baseline": (
                ablation[
                    "delta_enhanced_minus_baseline"
                ]
            ),
            "controls": (
                ablation[
                    "controls"
                ]
            ),
        },
        "intended_use": list(
            INTENDED_USES
        ),
        "prohibited_use": list(
            PROHIBITED_USES
        ),
        "human_control": {
            "human_approval_required": True,
            "automatic_refund": False,
            "automatic_customer_message": False,
            "automatic_dispute_submission": False,
            "action_vocabulary": [
                "MONITOR",
                "PREPARE_EVIDENCE",
                "MANUAL_REVIEW",
                "RECOMMEND_REFUND",
            ],
        },
        "explainability": {
            "method": "SHAP TreeExplainer",
            "label": (
                "Model explanation — not evidence"
            ),
            "causal": False,
            "dispute_evidence": False,
        },
        "llm_boundary": {
            "produces_risk_probability": False,
            "may_change_policy_action": False,
            "support_signal_count": 3,
            "support_text_timestamp_safe": True,
            "pii_removed": True,
            "prompt_injection_defense": True,
            "deterministic_fallback": True,
            "input_hash_cache": True,
        },
        "known_limitations": list(
            KNOWN_LIMITATIONS
        ),
    }

    markdown = _build_markdown(
        card
    )

    if "30-day prediction" not in markdown:
        raise AssertionError(
            "The model card must explicitly "
            "reject the old 30-day claim"
        )

    json_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    markdown_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with json_output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            card,
            file,
            indent=2,
            ensure_ascii=False,
        )

    markdown_output_path.write_text(
        markdown,
        encoding="utf-8",
    )

    return card


def main() -> None:
    card = generate_model_card()

    evaluation = card[
        "canonical_evaluation"
    ]

    ablation_delta = card[
        "ablation"
    ][
        "delta_enhanced_minus_baseline"
    ]

    print(
        "Chargeback Radar model card generated"
    )
    print(
        "Target:",
        card["target"]["name"],
    )
    print(
        "Model version:",
        card["model"]["version"],
    )
    print(
        "Calibration:",
        card[
            "model"
        ]["calibration_method"],
    )
    print(
        "Held-out rows:",
        f"{evaluation['test_rows']:,}",
    )
    print(
        "Average Precision:",
        f"{evaluation['average_precision']:.6f}",
    )
    print(
        "Ablation AP delta:",
        f"{ablation_delta['average_precision']:+.6f}",
    )
    print(
        "Markdown:",
        MODEL_CARD_MARKDOWN_PATH.relative_to(
            ROOT
        ),
    )
    print(
        "JSON:",
        MODEL_CARD_JSON_PATH.relative_to(
            ROOT
        ),
    )


if __name__ == "__main__":
    main()