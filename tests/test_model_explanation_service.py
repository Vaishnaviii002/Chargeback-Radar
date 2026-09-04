from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from pydantic import ValidationError

import src.model_explanation_service as service
from src.explanation_text import DISCLAIMER


EXPECTED_LABEL = "Model explanation — not evidence"
EXPECTED_TEXT_VERSION = "plain-v1"

FORBIDDEN_OUTPUT_FIELDS = {
    "customer_id",
    "customer_name",
    "customer_email",
    "customer_phone",
    "dispute_id",
    "dispute_status",
    "dispute_reason",
    "reason_code",
    "final_reason_code",
    "chargeback_within_120d",
    "chargeback_outcome",
    "true_fraud",
    "future_refund",
    "future_delivery",
}


def _positive_factors() -> list[dict[str, object]]:
    return [
        {
            "feature": "device_is_new",
            "value": 1,
            "shap_value": 0.42,
            "direction": "increases_risk",
        },
        {
            "feature": "txns_last_24h",
            "value": 5,
            "shap_value": 0.17,
            "direction": "increases_risk",
        },
    ]


def _negative_factors() -> list[dict[str, object]]:
    return [
        {
            "feature": "email_verified",
            "value": 1,
            "shap_value": -0.25,
            "direction": "decreases_risk",
        },
        {
            "feature": "phone_verified",
            "value": 1,
            "shap_value": -0.06,
            "direction": "decreases_risk",
        },
    ]


def _sample_row(
    *,
    payment_id: str = "pay_test_001",
    positive: list[dict[str, object]] | None = None,
    negative: list[dict[str, object]] | None = None,
) -> pd.Series:
    return pd.Series(
        {
            "payment_id": payment_id,
            "model_version": "0.1.0",
            "explanation_version": "shap-v1",
            "top_positive_factors_json": json.dumps(
                _positive_factors()
                if positive is None
                else positive
            ),
            "top_negative_factors_json": json.dumps(
                _negative_factors()
                if negative is None
                else negative
            ),
        }
    )


def _sample_report() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _sample_row(
                payment_id="pay_test_001"
            ).to_dict(),
            _sample_row(
                payment_id="pay_test_002"
            ).to_dict(),
        ]
    )


def test_required_production_reports_exist() -> None:
    assert service.SHAP_REPORT_PATH.exists(), (
        "Run python -m src.explain first"
    )

    assert service.DETERMINISTIC_REPORT_PATH.exists(), (
        "Run python -m src.model_explanation_service first"
    )


def test_production_report_matches_shap_report() -> None:
    shap_report = pd.read_parquet(
        service.SHAP_REPORT_PATH
    )

    text_report = pd.read_parquet(
        service.DETERMINISTIC_REPORT_PATH
    )

    assert len(text_report) == len(shap_report)
    assert not text_report.empty

    expected_ids = (
        shap_report["payment_id"]
        .astype(str)
        .tolist()
    )

    actual_ids = (
        text_report["payment_id"]
        .astype(str)
        .tolist()
    )

    assert actual_ids == expected_ids
    assert text_report["payment_id"].is_unique
    assert not text_report["payment_id"].isna().any()


def test_production_report_versions_are_exact() -> None:
    shap_report = pd.read_parquet(
        service.SHAP_REPORT_PATH
    )

    text_report = pd.read_parquet(
        service.DETERMINISTIC_REPORT_PATH
    )

    bundle_path = (
        service.ROOT
        / "artifacts"
        / "model_bundle.joblib"
    )

    bundle = joblib.load(bundle_path)

    expected_model_version = str(
        bundle["model_version"]
    )

    assert set(
        text_report["model_version"].astype(str)
    ) == {expected_model_version}

    assert (
        text_report["model_version"]
        .astype(str)
        .tolist()
        == shap_report["model_version"]
        .astype(str)
        .tolist()
    )

    assert (
        text_report["shap_explanation_version"]
        .astype(str)
        .tolist()
        == shap_report["explanation_version"]
        .astype(str)
        .tolist()
    )

    assert set(
        text_report["text_explanation_version"]
    ) == {EXPECTED_TEXT_VERSION}

    assert (
        service.TEXT_EXPLANATION_VERSION
        == EXPECTED_TEXT_VERSION
    )


def test_delivery_mode_label_and_disclaimer_are_exact() -> None:
    report = pd.read_parquet(
        service.DETERMINISTIC_REPORT_PATH
    )

    assert set(report["delivery_mode"]) == {
        "deterministic"
    }

    assert set(report["label"]) == {
        EXPECTED_LABEL
    }

    assert set(report["disclaimer"]) == {
        DISCLAIMER
    }

    assert report["explanation"].map(
        lambda text: DISCLAIMER in str(text)
    ).all()


def test_factor_payloads_match_shap_source_exactly() -> None:
    shap_report = pd.read_parquet(
        service.SHAP_REPORT_PATH
    )

    text_report = pd.read_parquet(
        service.DETERMINISTIC_REPORT_PATH
    )

    for shap_row, text_row in zip(
        shap_report.itertuples(index=False),
        text_report.itertuples(index=False),
        strict=True,
    ):
        expected_positive = json.loads(
            shap_row.top_positive_factors_json
        )

        expected_negative = json.loads(
            shap_row.top_negative_factors_json
        )

        actual_positive = json.loads(
            text_row.positive_factors_json
        )

        actual_negative = json.loads(
            text_row.negative_factors_json
        )

        assert actual_positive == expected_positive
        assert actual_negative == expected_negative


def test_no_forbidden_fields_are_introduced() -> None:
    report = pd.read_parquet(
        service.DETERMINISTIC_REPORT_PATH
    )

    normalized_columns = {
        str(column).strip().lower()
        for column in report.columns
    }

    assert normalized_columns.isdisjoint(
        FORBIDDEN_OUTPUT_FIELDS
    )

    for column in (
        "positive_factors_json",
        "negative_factors_json",
    ):
        for raw_payload in report[column]:
            factors = json.loads(raw_payload)

            for factor in factors:
                normalized_keys = {
                    str(key).strip().lower()
                    for key in factor
                }

                assert normalized_keys.isdisjoint(
                    FORBIDDEN_OUTPUT_FIELDS
                )


def test_build_explanation_preserves_factor_values() -> None:
    explanation = (
        service.build_model_explanation_from_row(
            _sample_row()
        )
    )

    assert explanation.payment_id == "pay_test_001"
    assert explanation.model_version == "0.1.0"
    assert (
        explanation.shap_explanation_version
        == "shap-v1"
    )
    assert (
        explanation.text_explanation_version
        == EXPECTED_TEXT_VERSION
    )
    assert explanation.delivery_mode == "deterministic"
    assert explanation.label == EXPECTED_LABEL
    assert explanation.disclaimer == DISCLAIMER

    assert [
        factor.model_dump()
        for factor in explanation.positive_factors
    ] == _positive_factors()

    assert [
        factor.model_dump()
        for factor in explanation.negative_factors
    ] == _negative_factors()


def test_malformed_factor_json_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Invalid factor JSON",
    ):
        service._parse_factors(
            "{not-valid-json"
        )


def test_non_list_factor_payload_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Factor payload must be a list",
    ):
        service._parse_factors(
            json.dumps(
                {
                    "feature": "device_is_new",
                }
            )
        )


def test_extra_factor_fields_are_rejected() -> None:
    payload = _positive_factors()[0].copy()
    payload["customer_id"] = "cust_forbidden"

    with pytest.raises(ValidationError):
        service.ExplanationFactor.model_validate(
            payload
        )


@pytest.mark.parametrize(
    ("shap_value", "direction"),
    [
        (0.25, "decreases_risk"),
        (-0.25, "increases_risk"),
        (0.0, "increases_risk"),
    ],
)
def test_direction_sign_mismatch_is_rejected(
    shap_value: float,
    direction: str,
) -> None:
    factor = service.ExplanationFactor(
        feature="device_is_new",
        value=1,
        shap_value=shap_value,
        direction=direction,
    )

    with pytest.raises(
        ValueError,
        match="Factor direction does not match",
    ):
        service._validate_factor_direction(
            factor
        )


def test_positive_factor_ordering_is_enforced() -> None:
    positive = _positive_factors()
    positive.reverse()

    with pytest.raises(
        ValueError,
        match="Positive factors are not ordered",
    ):
        service.build_model_explanation_from_row(
            _sample_row(
                positive=positive
            )
        )


def test_negative_factor_ordering_is_enforced() -> None:
    negative = _negative_factors()
    negative.reverse()

    with pytest.raises(
        ValueError,
        match="Negative factors are not ordered",
    ):
        service.build_model_explanation_from_row(
            _sample_row(
                negative=negative
            )
        )


def test_positive_list_rejects_non_positive_factor() -> None:
    positive = [
        {
            "feature": "device_is_new",
            "value": 1,
            "shap_value": 0.0,
            "direction": "neutral",
        }
    ]

    with pytest.raises(
        ValueError,
        match="non-positive contribution",
    ):
        service.build_model_explanation_from_row(
            _sample_row(
                positive=positive
            )
        )


def test_negative_list_rejects_non_negative_factor() -> None:
    negative = [
        {
            "feature": "email_verified",
            "value": 1,
            "shap_value": 0.0,
            "direction": "neutral",
        }
    ]

    with pytest.raises(
        ValueError,
        match="non-negative contribution",
    ):
        service.build_model_explanation_from_row(
            _sample_row(
                negative=negative
            )
        )


def test_unknown_payment_id_is_rejected(
    tmp_path: Path,
) -> None:
    report_path = (
        tmp_path
        / "shap_report.parquet"
    )

    _sample_report().to_parquet(
        report_path,
        index=False,
    )

    with pytest.raises(
        KeyError,
        match="Unknown held-out payment_id",
    ):
        service.get_model_explanation(
            "pay_unknown",
            report_path=report_path,
        )


def test_empty_payment_id_is_rejected(
    tmp_path: Path,
) -> None:
    report_path = (
        tmp_path
        / "shap_report.parquet"
    )

    _sample_report().to_parquet(
        report_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="payment_id cannot be empty",
    ):
        service.get_model_explanation(
            "",
            report_path=report_path,
        )


def test_duplicate_payment_ids_are_rejected(
    tmp_path: Path,
) -> None:
    report = _sample_report()

    report.loc[
        1,
        "payment_id",
    ] = report.loc[
        0,
        "payment_id",
    ]

    report_path = (
        tmp_path
        / "duplicate.parquet"
    )

    report.to_parquet(
        report_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="duplicate payment IDs",
    ):
        service._load_report(
            report_path
        )


def test_missing_required_columns_are_rejected(
    tmp_path: Path,
) -> None:
    report = _sample_report().drop(
        columns=[
            "top_negative_factors_json"
        ]
    )

    report_path = (
        tmp_path
        / "missing_column.parquet"
    )

    report.to_parquet(
        report_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        service._load_report(
            report_path
        )


def test_repeated_generation_is_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _sample_report()

    def fake_load_report(
        *_args: object,
        **_kwargs: object,
    ) -> pd.DataFrame:
        return source.copy(deep=True)

    monkeypatch.setattr(
        service,
        "_load_report",
        fake_load_report,
    )

    first_path = (
        tmp_path
        / "first.parquet"
    )

    second_path = (
        tmp_path
        / "second.parquet"
    )

    first = (
        service.generate_deterministic_report(
            output_path=first_path
        )
    )

    second = (
        service.generate_deterministic_report(
            output_path=second_path
        )
    )

    pd.testing.assert_frame_equal(
        first,
        second,
        check_exact=True,
    )

    pd.testing.assert_frame_equal(
        pd.read_parquet(first_path),
        pd.read_parquet(second_path),
        check_exact=True,
    )