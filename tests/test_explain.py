from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

import src.explain as explain


def _load_report() -> pd.DataFrame:
    assert explain.OUTPUT_PATH.exists(), (
        "Run `python -m src.explain` before these tests."
    )

    return pd.read_parquet(explain.OUTPUT_PATH)


def _load_factors(
    value: str,
) -> list[dict]:
    parsed = json.loads(value)

    assert isinstance(parsed, list)

    return parsed


def test_explanation_report_exists() -> None:
    assert explain.OUTPUT_PATH.exists()


def test_explanation_row_count_matches_heldout() -> None:
    report = _load_report()

    heldout = pd.read_parquet(
        explain.HELDOUT_REPORT_PATH
    )

    assert len(report) == len(heldout)


def test_explanation_payment_ids_align_exactly() -> None:
    report = _load_report()

    heldout = pd.read_parquet(
        explain.HELDOUT_REPORT_PATH
    )

    expected = (
        heldout["payment_id"]
        .astype(str)
        .tolist()
    )

    actual = (
        report["payment_id"]
        .astype(str)
        .tolist()
    )

    assert actual == expected


def test_explanation_payment_ids_are_unique() -> None:
    report = _load_report()

    assert not report[
        "payment_id"
    ].duplicated().any()


def test_customer_identity_not_in_output_schema() -> None:
    report = _load_report()

    forbidden_columns = {
        "customer_id",
        "dispute_id",
    }

    assert forbidden_columns.isdisjoint(
        report.columns
    )


def test_model_version_matches_exact_bundle() -> None:
    report = _load_report()

    bundle = joblib.load(
        explain.MODEL_BUNDLE_PATH
    )

    expected_version = str(
        bundle["model_version"]
    )

    assert set(
        report["model_version"].astype(str)
    ) == {
        expected_version
    }


def test_target_matches_exact_trained_bundle() -> None:
    report = _load_report()

    bundle = joblib.load(
        explain.MODEL_BUNDLE_PATH
    )

    expected_target = str(
        bundle["target"]
    )

    assert set(
        report["target"].astype(str)
    ) == {
        expected_target
    }


def test_explanation_version_present() -> None:
    report = _load_report()

    assert set(
        report[
            "explanation_version"
        ].astype(str)
    ) == {
        explain.EXPLANATION_VERSION
    }


def test_feature_source_is_authoritative() -> None:
    report = _load_report()

    assert set(
        report["feature_source"]
    ) == {
        "features.parquet"
    }


def test_no_forbidden_factor_names() -> None:
    report = _load_report()

    forbidden = (
        explain.IDENTIFIER_FIELDS
        | explain.FORBIDDEN_MODEL_FIELDS
    )

    json_columns = [
        "top_positive_factors_json",
        "top_negative_factors_json",
        "all_feature_contributions_json",
    ]

    for column in json_columns:
        for raw in report[column]:
            factors = _load_factors(raw)

            for factor in factors:
                feature = factor["feature"]

                assert feature not in forbidden


def test_only_exact_model_features_appear() -> None:
    report = _load_report()

    bundle = joblib.load(
        explain.MODEL_BUNDLE_PATH
    )

    allowed = set(
        bundle["model_features"]
    )

    for raw in report[
        "all_feature_contributions_json"
    ]:
        factors = _load_factors(raw)

        observed = {
            factor["feature"]
            for factor in factors
        }

        assert observed == allowed


def test_positive_factors_are_correctly_ordered() -> None:
    report = _load_report()

    for raw in report[
        "top_positive_factors_json"
    ]:
        factors = _load_factors(raw)

        values = [
            float(
                factor["shap_value"]
            )
            for factor in factors
        ]

        assert all(
            value > 0
            for value in values
        )

        assert values == sorted(
            values,
            reverse=True,
        )


def test_negative_factors_are_correctly_ordered() -> None:
    report = _load_report()

    for raw in report[
        "top_negative_factors_json"
    ]:
        factors = _load_factors(raw)

        values = [
            float(
                factor["shap_value"]
            )
            for factor in factors
        ]

        assert all(
            value < 0
            for value in values
        )

        assert values == sorted(
            values
        )


def test_factor_direction_matches_shap_sign() -> None:
    report = _load_report()

    for raw in report[
        "all_feature_contributions_json"
    ]:
        factors = _load_factors(raw)

        for factor in factors:
            value = float(
                factor["shap_value"]
            )

            direction = factor[
                "direction"
            ]

            if value > 0:
                assert (
                    direction
                    == "increases_risk"
                )

            elif value < 0:
                assert (
                    direction
                    == "decreases_risk"
                )

            else:
                assert (
                    direction
                    == "neutral"
                )


def test_all_factors_sorted_by_absolute_contribution() -> None:
    report = _load_report()

    for raw in report[
        "all_feature_contributions_json"
    ]:
        factors = _load_factors(raw)

        magnitudes = [
            abs(
                float(
                    factor[
                        "shap_value"
                    ]
                )
            )
            for factor in factors
        ]

        assert magnitudes == sorted(
            magnitudes,
            reverse=True,
        )


def test_shap_reconstruction_error_is_small() -> None:
    report = _load_report()

    if (
        "shap_reconstruction_error"
        not in report.columns
    ):
        return

    assert (
        report[
            "shap_reconstruction_error"
        ].max()
        <= 1e-4
    )


def test_deterministic_shap_values_on_same_transactions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    all_ids = explain._load_heldout_ids()

    sample_ids = all_ids[:32]

    monkeypatch.setattr(
        explain,
        "_load_heldout_ids",
        lambda: sample_ids,
    )

    first_path = (
        tmp_path
        / "first.parquet"
    )

    second_path = (
        tmp_path
        / "second.parquet"
    )

    first = explain.generate_explanations(
        output_path=first_path
    )

    second = explain.generate_explanations(
        output_path=second_path
    )

    deterministic_columns = [
        "payment_id",
        "heldout_row_position",
        "model_version",
        "explanation_version",
        "target",
        "heldout_report",
        "feature_source",
        "base_value_raw",
        "shap_reconstructed_raw_score",
        "top_positive_factors_json",
        "top_negative_factors_json",
        "all_feature_contributions_json",
    ]

    if (
        "model_raw_score"
        in first.columns
    ):
        deterministic_columns.append(
            "model_raw_score"
        )

    if (
        "shap_reconstruction_error"
        in first.columns
    ):
        deterministic_columns.append(
            "shap_reconstruction_error"
        )

    pd.testing.assert_frame_equal(
        first[
            deterministic_columns
        ].reset_index(drop=True),
        second[
            deterministic_columns
        ].reset_index(drop=True),
        check_exact=True,
    )