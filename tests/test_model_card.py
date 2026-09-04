from __future__ import annotations

import json
from pathlib import Path
import re
import joblib
import pandas as pd
import pytest

from src.model_card import (
    AUTHORITATIVE_TARGET,
    MODEL_BUNDLE_PATH,
    MODEL_CARD_JSON_PATH,
    MODEL_CARD_MARKDOWN_PATH,
    MODEL_CARD_VERSION,
    generate_model_card,
)


def _load_json(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def test_model_card_outputs_exist() -> None:
    assert MODEL_CARD_JSON_PATH.exists(), (
        "Run python -m src.model_card first"
    )

    assert (
        MODEL_CARD_MARKDOWN_PATH.exists()
    ), (
        "Run python -m src.model_card first"
    )


def test_target_and_model_identity_are_exact() -> None:
    card = _load_json(
        MODEL_CARD_JSON_PATH
    )

    bundle = joblib.load(
        MODEL_BUNDLE_PATH
    )

    assert (
        card["model_card_version"]
        == MODEL_CARD_VERSION
    )

    assert (
        card["target"]["name"]
        == AUTHORITATIVE_TARGET
    )

    assert (
        card["target"]["window_days"]
        == 120
    )

    assert (
        card["model"]["version"]
        == str(
            bundle["model_version"]
        )
    )


def test_features_match_saved_model_bundle() -> None:
    card = _load_json(
        MODEL_CARD_JSON_PATH
    )

    bundle = joblib.load(
        MODEL_BUNDLE_PATH
    )

    assert (
        card["features"][
            "model_features"
        ]
        == bundle["model_features"]
    )

    assert (
        card["features"][
            "categorical_features"
        ]
        == bundle[
            "categorical_features"
        ]
    )

    assert (
        card["features"][
            "numerical_features"
        ]
        == bundle[
            "numerical_features"
        ]
    )


def test_calibration_method_is_source_backed() -> None:
    card = _load_json(
        MODEL_CARD_JSON_PATH
    )

    calibration = _load_json(
        Path(
            "reports/calibration_metrics.json"
        )
    )

    assert (
        card["model"][
            "calibration_method"
        ]
        == calibration[
            "selected_method"
        ]
    )


def test_canonical_metrics_are_source_backed() -> None:
    card = _load_json(
        MODEL_CARD_JSON_PATH
    )

    metrics = _load_json(
        Path(
            "reports/metrics.json"
        )
    )

    calibration = _load_json(
        Path(
            "reports/calibration_metrics.json"
        )
    )

    evaluation = card[
        "canonical_evaluation"
    ]

    performance = metrics[
        "model_performance"
    ]

    assert (
        evaluation["test_rows"]
        == metrics[
            "dataset"
        ]["test_rows"]
    )

    assert (
        evaluation["positive_labels"]
        == metrics[
            "dataset"
        ]["positive_labels"]
    )

    assert (
        evaluation["average_precision"]
        == pytest.approx(
            performance[
                "average_precision"
            ]
        )
    )

    assert (
        evaluation["precision"]
        == pytest.approx(
            performance["precision"]
        )
    )

    assert (
        evaluation["recall"]
        == pytest.approx(
            performance["recall"]
        )
    )

    assert (
        evaluation["brier_score"]
        == pytest.approx(
            calibration[
                "test_calibrated_brier"
            ]
        )
    )


def test_unseen_customer_claim_is_supported() -> None:
    card = _load_json(
        MODEL_CARD_JSON_PATH
    )

    metadata = _load_json(
        Path(
            "artifacts/model_metadata.json"
        )
    )

    stress = card[
        "unseen_customer_stress_test"
    ]

    assert (
        metadata["customer_disjoint"]
        is True
    )

    assert (
        stress["unique_test_customers"]
        == metadata[
            "test_unique_customers"
        ]
    )

    assert stress[
        "training_overlap"
    ] == 0

    assert stress[
        "calibration_overlap"
    ] == 0


def test_ablation_is_copied_without_cherry_picking() -> None:
    card = _load_json(
        MODEL_CARD_JSON_PATH
    )

    ablation = _load_json(
        Path(
            "reports/ablation.json"
        )
    )

    assert (
        card["ablation"][
            "baseline_metrics"
        ]
        == ablation[
            "baseline"
        ]["metrics"]
    )

    assert (
        card["ablation"][
            "enhanced_metrics"
        ]
        == ablation[
            "enhanced"
        ]["metrics"]
    )

    assert (
        card["ablation"][
            "delta_enhanced_minus_baseline"
        ]
        == ablation[
            "delta_enhanced_minus_baseline"
        ]
    )

    assert (
        card["ablation"][
            "delta_enhanced_minus_baseline"
        ]["precision"]
        < 0
    )


def test_support_summary_matches_report() -> None:
    card = _load_json(
        MODEL_CARD_JSON_PATH
    )

    support_report = pd.read_parquet(
        "reports/support_signals.parquet"
    )

    summary = card[
        "support_text_signals"
    ]

    assert (
        summary["rows"]
        == len(support_report)
    )

    assert (
        summary[
            "unique_sanitized_input_hashes"
        ]
        == support_report[
            "input_hash"
        ].nunique()
    )

    assert (
        summary[
            "payments_with_observable_text"
        ]
        == (
            support_report[
                "observable_event_count"
            ]
            > 0
        ).sum()
    )

    assert (
        summary[
            "excluded_future_events"
        ]
        == support_report[
            "excluded_future_event_count"
        ].sum()
    )


def test_human_controls_are_enforced() -> None:
    card = _load_json(
        MODEL_CARD_JSON_PATH
    )

    controls = card[
        "human_control"
    ]

    assert (
        controls[
            "human_approval_required"
        ]
        is True
    )

    assert (
        controls[
            "automatic_refund"
        ]
        is False
    )

    assert (
        controls[
            "automatic_customer_message"
        ]
        is False
    )

    assert (
        controls[
            "automatic_dispute_submission"
        ]
        is False
    )


def test_markdown_contains_required_disclosures() -> None:
    markdown = (
        MODEL_CARD_MARKDOWN_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    assert (
        "chargeback_within_120d"
        in markdown
    )

    assert (
        "This is a 120-day target"
        in markdown
    )

    assert (
        "must not be described as a "
        "30-day prediction"
        in markdown
    )

    assert (
        "Model explanation — not evidence"
        in markdown
    )

    assert (
        "synthetic"
        in markdown.lower()
    )

    assert (
        "human approval"
        in markdown.lower()
    )

    assert (
        "does not establish real-world"
        in markdown
    )


def test_outputs_contain_no_secrets() -> None:
    combined = (
        MODEL_CARD_MARKDOWN_PATH
        .read_text(
            encoding="utf-8"
        )
        + MODEL_CARD_JSON_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    assert "OPENAI_API_KEY=" not in combined
    assert "RAZORPAY_KEY_SECRET=" not in combined
    assert "rzp_test_" not in combined
    assert not re.search(
    r"\bsk-[A-Za-z0-9_-]{16,}\b",
    combined,
)


def test_generation_is_deterministic(
    tmp_path: Path,
) -> None:
    first_json = (
        tmp_path
        / "first.json"
    )

    first_markdown = (
        tmp_path
        / "first.md"
    )

    second_json = (
        tmp_path
        / "second.json"
    )

    second_markdown = (
        tmp_path
        / "second.md"
    )

    first = generate_model_card(
        json_output_path=first_json,
        markdown_output_path=(
            first_markdown
        ),
    )

    second = generate_model_card(
        json_output_path=second_json,
        markdown_output_path=(
            second_markdown
        ),
    )

    assert first == second

    assert (
        first_json.read_text(
            encoding="utf-8"
        )
        == second_json.read_text(
            encoding="utf-8"
        )
    )

    assert (
        first_markdown.read_text(
            encoding="utf-8"
        )
        == second_markdown.read_text(
            encoding="utf-8"
        )
    )