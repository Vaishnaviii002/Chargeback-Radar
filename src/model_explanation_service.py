from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.explanation_text import (
    DISCLAIMER,
    build_deterministic_explanation,
)


ROOT = Path(__file__).resolve().parents[1]

SHAP_REPORT_PATH = (
    ROOT
    / "reports"
    / "test_explanations.parquet"
)

DETERMINISTIC_REPORT_PATH = (
    ROOT
    / "reports"
    / "test_model_explanations.parquet"
)

TEXT_EXPLANATION_VERSION = "plain-v1"


Direction = Literal[
    "increases_risk",
    "decreases_risk",
    "neutral",
]


class ExplanationFactor(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    feature: str = Field(
        min_length=1,
    )

    value: Any

    shap_value: float

    direction: Direction


class ModelExplanation(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    payment_id: str = Field(
        min_length=1,
    )

    model_version: str = Field(
        min_length=1,
    )

    shap_explanation_version: str = Field(
        min_length=1,
    )

    text_explanation_version: str = Field(
        min_length=1,
    )

    delivery_mode: Literal[
        "deterministic"
    ]

    label: Literal[
        "Model explanation — not evidence"
    ]

    explanation: str = Field(
        min_length=1,
    )

    disclaimer: str = Field(
        min_length=1,
    )

    positive_factors: list[
        ExplanationFactor
    ]

    negative_factors: list[
        ExplanationFactor
    ]


def _load_report(
    report_path: Path = SHAP_REPORT_PATH,
) -> pd.DataFrame:
    if not report_path.exists():
        raise FileNotFoundError(
            f"SHAP report is missing: {report_path}"
        )

    report = pd.read_parquet(
        report_path
    )

    required_columns = {
        "payment_id",
        "model_version",
        "explanation_version",
        "top_positive_factors_json",
        "top_negative_factors_json",
    }

    missing = sorted(
        required_columns
        - set(report.columns)
    )

    if missing:
        raise ValueError(
            "SHAP report is missing required columns: "
            + ", ".join(missing)
        )

    if report.empty:
        raise ValueError(
            "SHAP report contains zero rows"
        )

    if report[
        "payment_id"
    ].isna().any():
        raise ValueError(
            "SHAP report contains missing payment_id"
        )

    report = report.copy()

    report["payment_id"] = (
        report["payment_id"]
        .astype(str)
    )

    if report[
        "payment_id"
    ].duplicated().any():
        raise ValueError(
            "SHAP report contains duplicate payment IDs"
        )

    return report


def _parse_factors(
    raw: str,
) -> list[ExplanationFactor]:
    try:
        value = json.loads(
            raw
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Invalid factor JSON"
        ) from exc

    if not isinstance(
        value,
        list,
    ):
        raise ValueError(
            "Factor payload must be a list"
        )

    return [
        ExplanationFactor.model_validate(
            factor
        )
        for factor in value
    ]


def _validate_factor_direction(
    factor: ExplanationFactor,
) -> None:
    if factor.shap_value > 0:
        expected = "increases_risk"

    elif factor.shap_value < 0:
        expected = "decreases_risk"

    else:
        expected = "neutral"

    if factor.direction != expected:
        raise ValueError(
            "Factor direction does not match "
            f"SHAP sign for {factor.feature}"
        )


def _validate_factor_sets(
    positive: list[ExplanationFactor],
    negative: list[ExplanationFactor],
) -> None:
    for factor in positive:
        _validate_factor_direction(
            factor
        )

        if factor.shap_value <= 0:
            raise ValueError(
                "Positive-factor list contains "
                "a non-positive contribution"
            )

    for factor in negative:
        _validate_factor_direction(
            factor
        )

        if factor.shap_value >= 0:
            raise ValueError(
                "Negative-factor list contains "
                "a non-negative contribution"
            )

    positive_values = [
        factor.shap_value
        for factor in positive
    ]

    if positive_values != sorted(
        positive_values,
        reverse=True,
    ):
        raise ValueError(
            "Positive factors are not "
            "ordered by contribution"
        )

    negative_values = [
        factor.shap_value
        for factor in negative
    ]

    if negative_values != sorted(
        negative_values
    ):
        raise ValueError(
            "Negative factors are not "
            "ordered by contribution"
        )


def _factor_dicts(
    factors: list[ExplanationFactor],
) -> list[dict[str, Any]]:
    return [
        factor.model_dump()
        for factor in factors
    ]


def build_model_explanation_from_row(
    row: pd.Series,
) -> ModelExplanation:
    positive = _parse_factors(
        str(
            row[
                "top_positive_factors_json"
            ]
        )
    )

    negative = _parse_factors(
        str(
            row[
                "top_negative_factors_json"
            ]
        )
    )

    _validate_factor_sets(
        positive,
        negative,
    )

    positive_dicts = _factor_dicts(
        positive
    )

    negative_dicts = _factor_dicts(
        negative
    )

    text = build_deterministic_explanation(
        positive_factors=positive_dicts,
        negative_factors=negative_dicts,
        max_positive=3,
        max_negative=2,
    )

    if DISCLAIMER not in text:
        raise AssertionError(
            "Model explanation is missing "
            "the required disclaimer"
        )

    return ModelExplanation(
        payment_id=str(
            row["payment_id"]
        ),
        model_version=str(
            row["model_version"]
        ),
        shap_explanation_version=str(
            row["explanation_version"]
        ),
        text_explanation_version=(
            TEXT_EXPLANATION_VERSION
        ),
        delivery_mode="deterministic",
        label=(
            "Model explanation — not evidence"
        ),
        explanation=text,
        disclaimer=DISCLAIMER,
        positive_factors=positive,
        negative_factors=negative,
    )


def get_model_explanation(
    payment_id: str,
    report_path: Path = SHAP_REPORT_PATH,
) -> ModelExplanation:
    if not payment_id:
        raise ValueError(
            "payment_id cannot be empty"
        )

    report = _load_report(
        report_path
    )

    matches = report.loc[
        report["payment_id"]
        == str(payment_id)
    ]

    if matches.empty:
        raise KeyError(
            f"Unknown held-out payment_id: "
            f"{payment_id}"
        )

    if len(matches) != 1:
        raise ValueError(
            "payment_id matched more than "
            "one SHAP explanation row"
        )

    return build_model_explanation_from_row(
        matches.iloc[0]
    )


def generate_deterministic_report(
    output_path: Path = DETERMINISTIC_REPORT_PATH,
) -> pd.DataFrame:
    report = _load_report()

    records: list[
        dict[str, Any]
    ] = []

    for _, row in report.iterrows():
        explanation = (
            build_model_explanation_from_row(
                row
            )
        )

        records.append(
            {
                "payment_id": (
                    explanation.payment_id
                ),
                "model_version": (
                    explanation.model_version
                ),
                "shap_explanation_version": (
                    explanation.shap_explanation_version
                ),
                "text_explanation_version": (
                    explanation.text_explanation_version
                ),
                "delivery_mode": (
                    explanation.delivery_mode
                ),
                "label": explanation.label,
                "explanation": (
                    explanation.explanation
                ),
                "disclaimer": (
                    explanation.disclaimer
                ),
                "positive_factors_json": (
                    json.dumps(
                        [
                            factor.model_dump()
                            for factor
                            in explanation.positive_factors
                        ],
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                ),
                "negative_factors_json": (
                    json.dumps(
                        [
                            factor.model_dump()
                            for factor
                            in explanation.negative_factors
                        ],
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                ),
            }
        )

    result = pd.DataFrame.from_records(
        records
    )

    expected_ids = (
        report["payment_id"]
        .astype(str)
        .tolist()
    )

    actual_ids = (
        result["payment_id"]
        .astype(str)
        .tolist()
    )

    if actual_ids != expected_ids:
        raise AssertionError(
            "Text explanations are not "
            "aligned with SHAP rows"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        output_path,
        index=False,
    )

    return result


def main() -> None:
    result = (
        generate_deterministic_report()
    )

    print(
        "Deterministic model explanations generated"
    )

    print(
        f"Rows: {len(result):,}"
    )

    print(
        "Output:",
        DETERMINISTIC_REPORT_PATH.relative_to(
            ROOT
        ),
    )

    print(
        "Text explanation version:",
        TEXT_EXPLANATION_VERSION,
    )


if __name__ == "__main__":
    main()