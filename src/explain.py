from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap


ROOT = Path(__file__).resolve().parents[1]

MODEL_BUNDLE_PATH = ROOT / "artifacts" / "model_bundle.joblib"
FEATURE_SCHEMA_PATH = ROOT / "artifacts" / "feature_schema.json"
MODEL_METADATA_PATH = ROOT / "artifacts" / "model_metadata.json"

FEATURE_SOURCE_PATH = ROOT / "data" / "features.parquet"
HELDOUT_REPORT_PATH = ROOT / "reports" / "test_scored.parquet"

OUTPUT_PATH = ROOT / "reports" / "test_explanations.parquet"

EXPLANATION_VERSION = "shap-v1"
TOP_K = 5


IDENTIFIER_FIELDS = {
    "payment_id",
    "customer_id",
    "dispute_id",
}


FORBIDDEN_MODEL_FIELDS = {
    "chargeback_within_120d",
    "chargeback",
    "is_chargeback",
    "chargeback_status",
    "chargeback_reason",
    "chargeback_reason_code",
    "reason_code",
    "reason_family",
    "dispute_status",
    "dispute_reason",
    "final_dispute_status",
    "refund_status",
    "refund_completed",
    "refund_completed_at",
    "delivery_status",
    "delivered_at",
    "shipment_delivered",
    "is_duplicate_payment",
    "cancelled_subscription_billed",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file is missing: {path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)

    if not isinstance(value, dict):
        raise TypeError(
            f"{path} must contain a JSON object"
        )

    return value


def _load_bundle() -> dict[str, Any]:
    if not MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"Model bundle is missing: {MODEL_BUNDLE_PATH}"
        )

    bundle = joblib.load(MODEL_BUNDLE_PATH)

    if not isinstance(bundle, dict):
        raise TypeError(
            "model_bundle.joblib must contain a dictionary"
        )

    required_keys = {
        "model",
        "preprocessor",
        "model_features",
        "categorical_features",
        "numerical_features",
        "target",
        "model_version",
    }

    missing = sorted(
        required_keys - set(bundle.keys())
    )

    if missing:
        raise ValueError(
            "Model bundle is missing required keys: "
            + ", ".join(missing)
        )

    return bundle


def _validate_contract(
    bundle: dict[str, Any],
    schema: dict[str, Any],
    metadata: dict[str, Any],
) -> list[str]:
    model_features = [
        str(feature)
        for feature in bundle["model_features"]
    ]

    schema_features = [
        str(feature)
        for feature in schema.get("model_features", [])
    ]

    if not model_features:
        raise ValueError("Model feature list is empty")

    if len(model_features) != len(set(model_features)):
        raise ValueError(
            "Model feature list contains duplicates"
        )

    if model_features != schema_features:
        raise ValueError(
            "Model bundle feature order does not exactly match "
            "feature_schema.json"
        )

    bundle_target = str(bundle["target"])
    schema_target = str(schema.get("target"))

    if bundle_target != schema_target:
        raise ValueError(
            "Model bundle target does not match feature schema"
        )

    bundle_version = str(bundle["model_version"])
    metadata_version = str(metadata.get("model_version"))

    if bundle_version != metadata_version:
        raise ValueError(
            "Model bundle version does not match model metadata"
        )

    forbidden = (
        IDENTIFIER_FIELDS
        | FORBIDDEN_MODEL_FIELDS
        | {bundle_target}
    )

    leaked = sorted(
        set(model_features) & forbidden
    )

    if leaked:
        raise ValueError(
            "Forbidden fields reached model inputs: "
            + ", ".join(leaked)
        )

    return model_features


def _load_heldout_ids() -> list[str]:
    if not HELDOUT_REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Held-out report missing: {HELDOUT_REPORT_PATH}"
        )

    heldout = pd.read_parquet(
        HELDOUT_REPORT_PATH
    )

    if "payment_id" not in heldout.columns:
        raise ValueError(
            "test_scored.parquet does not contain payment_id"
        )

    if heldout.empty:
        raise ValueError(
            "Held-out report contains zero rows"
        )

    if heldout["payment_id"].isna().any():
        raise ValueError(
            "Held-out report contains missing payment_id"
        )

    ids = (
        heldout["payment_id"]
        .astype(str)
        .tolist()
    )

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Held-out report contains duplicate payment IDs"
        )

    return ids


def _load_aligned_features(
    heldout_ids: list[str],
    model_features: list[str],
) -> pd.DataFrame:
    if not FEATURE_SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Feature source missing: {FEATURE_SOURCE_PATH}"
        )

    feature_frame = pd.read_parquet(
        FEATURE_SOURCE_PATH
    )

    required_columns = {
        "payment_id",
        *model_features,
    }

    missing = sorted(
        required_columns - set(feature_frame.columns)
    )

    if missing:
        raise ValueError(
            "data/features.parquet is missing model columns: "
            + ", ".join(missing)
        )

    feature_frame = feature_frame.loc[
        :,
        ["payment_id", *model_features],
    ].copy()

    if feature_frame["payment_id"].isna().any():
        raise ValueError(
            "Feature source contains missing payment_id"
        )

    feature_frame["payment_id"] = (
        feature_frame["payment_id"].astype(str)
    )

    if feature_frame["payment_id"].duplicated().any():
        raise ValueError(
            "Feature source contains duplicate payment IDs"
        )

    indexed = feature_frame.set_index(
        "payment_id",
        drop=False,
    )

    missing_ids = [
        payment_id
        for payment_id in heldout_ids
        if payment_id not in indexed.index
    ]

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} held-out payment IDs are missing "
            f"from data/features.parquet. "
            f"Examples: {missing_ids[:5]}"
        )

    aligned = (
        indexed
        .loc[heldout_ids]
        .reset_index(drop=True)
    )

    aligned_ids = (
        aligned["payment_id"]
        .astype(str)
        .tolist()
    )

    if aligned_ids != heldout_ids:
        raise AssertionError(
            "Feature rows are not aligned to held-out payment IDs"
        )

    if len(aligned) != len(heldout_ids):
        raise AssertionError(
            "Held-out row count changed during feature alignment"
        )

    return aligned


def _to_dense(value: Any) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()

    array = np.asarray(value)

    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2-D transformed matrix, got {array.shape}"
        )

    return array


def _transformed_feature_names(
    bundle: dict[str, Any],
    transformed: np.ndarray,
) -> list[str]:
    stored = bundle.get(
        "transformed_feature_names"
    )

    if stored is not None:
        names = [
            str(name)
            for name in list(stored)
        ]

    else:
        preprocessor = bundle["preprocessor"]

        if not hasattr(
            preprocessor,
            "get_feature_names_out",
        ):
            raise ValueError(
                "No transformed feature names are available"
            )

        names = [
            str(name)
            for name in preprocessor.get_feature_names_out()
        ]

    if len(names) != transformed.shape[1]:
        raise ValueError(
            "Transformed feature-name count does not match "
            "transformed matrix width"
        )

    if len(names) != len(set(names)):
        raise ValueError(
            "Transformed feature names contain duplicates"
        )

    return names


def _source_feature_name(
    transformed_name: str,
    model_features: list[str],
) -> str:
    if "__" in transformed_name:
        core = transformed_name.split("__", 1)[1]
    else:
        core = transformed_name

    ordered = sorted(
        model_features,
        key=len,
        reverse=True,
    )

    for feature in ordered:
        if core == feature:
            return feature

        for separator in ("_", "=", "["):
            if core.startswith(
                feature + separator
            ):
                return feature

    raise ValueError(
        "Could not map transformed feature back to "
        f"an original model feature: {transformed_name}"
    )


def _normalise_shap_values(
    values: Any,
    rows: int,
    features: int,
) -> np.ndarray:
    if isinstance(values, list):
        if not values:
            raise ValueError(
                "SHAP returned an empty list"
            )

        values = values[-1]

    array = np.asarray(values)

    if array.ndim == 2:
        if array.shape != (rows, features):
            raise ValueError(
                f"Unexpected SHAP shape {array.shape}; "
                f"expected {(rows, features)}"
            )

        return array.astype(float)

    if (
        array.ndim == 3
        and array.shape[0] == rows
        and array.shape[1] == features
    ):
        return array[:, :, -1].astype(float)

    if (
        array.ndim == 3
        and array.shape[0] == rows
        and array.shape[2] == features
    ):
        return array[:, -1, :].astype(float)

    raise ValueError(
        f"Unsupported SHAP output shape: {array.shape}"
    )


def _positive_expected_value(value: Any) -> float:
    array = np.asarray(value)

    if array.ndim == 0:
        return float(array)

    flat = array.reshape(-1)

    if len(flat) == 0:
        raise ValueError(
            "SHAP expected value is empty"
        )

    return float(flat[-1])


def _safe_value(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if np.isnan(value):
            return None

        return float(value)

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None

        return value.isoformat()

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (str, int, float)):
        return value

    return str(value)


def _factor(
    feature: str,
    raw_value: Any,
    contribution: float,
) -> dict[str, Any]:
    if contribution > 0:
        direction = "increases_risk"
    elif contribution < 0:
        direction = "decreases_risk"
    else:
        direction = "neutral"

    return {
        "feature": feature,
        "value": _safe_value(raw_value),
        "shap_value": float(contribution),
        "direction": direction,
    }


def _raw_model_scores(
    model: Any,
    transformed: np.ndarray,
) -> np.ndarray | None:
    try:
        raw = model.predict(
            transformed,
            raw_score=True,
        )
    except (
        TypeError,
        ValueError,
        AttributeError,
    ):
        return None

    raw = np.asarray(
        raw,
        dtype=float,
    ).reshape(-1)

    if len(raw) != len(transformed):
        raise ValueError(
            "Raw-score prediction row count does not match "
            "the transformed matrix"
        )

    return raw


def generate_explanations(
    output_path: Path = OUTPUT_PATH,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    if top_k < 1:
        raise ValueError(
            "top_k must be at least 1"
        )

    bundle = _load_bundle()

    schema = _load_json(
        FEATURE_SCHEMA_PATH
    )

    metadata = _load_json(
        MODEL_METADATA_PATH
    )

    model_features = _validate_contract(
        bundle,
        schema,
        metadata,
    )

    heldout_ids = _load_heldout_ids()

    feature_frame = _load_aligned_features(
        heldout_ids,
        model_features,
    )

    raw_features = feature_frame[
        model_features
    ].copy()

    preprocessor = bundle[
        "preprocessor"
    ]

    transformed = _to_dense(
        preprocessor.transform(
            raw_features
        )
    )

    if transformed.shape[0] != len(feature_frame):
        raise AssertionError(
            "Preprocessing changed the number of held-out rows"
        )

    transformed_names = _transformed_feature_names(
        bundle,
        transformed,
    )

    source_mapping = [
        _source_feature_name(
            name,
            model_features,
        )
        for name in transformed_names
    ]

    model = bundle[
        "model"
    ]

    explainer = shap.TreeExplainer(
        model
    )

    raw_shap_values = explainer.shap_values(
        transformed,
        check_additivity=False,
    )

    shap_values = _normalise_shap_values(
        raw_shap_values,
        rows=transformed.shape[0],
        features=transformed.shape[1],
    )

    if not np.isfinite(
        shap_values
    ).all():
        raise ValueError(
            "SHAP values contain NaN or infinite values"
        )

    base_value = _positive_expected_value(
        explainer.expected_value
    )

    raw_scores = _raw_model_scores(
        model,
        transformed,
    )

    reconstructed_raw = (
        base_value
        + shap_values.sum(axis=1)
    )

    reconstruction_error: np.ndarray | None = None

    if raw_scores is not None:
        reconstruction_error = np.abs(
            reconstructed_raw
            - raw_scores
        )

        max_error = float(
            reconstruction_error.max()
        )

        if max_error > 1e-4:
            raise ValueError(
                "SHAP additivity validation failed. "
                f"Maximum raw-score reconstruction error: "
                f"{max_error}"
            )

    generated_at = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )

    records: list[dict[str, Any]] = []

    for row_index in range(
        len(feature_frame)
    ):
        aggregated = {
            feature: 0.0
            for feature in model_features
        }

        for contribution, source_feature in zip(
            shap_values[row_index],
            source_mapping,
            strict=True,
        ):
            aggregated[source_feature] += float(
                contribution
            )

        raw_row = raw_features.iloc[
            row_index
        ]

        factors = [
            _factor(
                feature=feature,
                raw_value=raw_row[feature],
                contribution=contribution,
            )
            for feature, contribution
            in aggregated.items()
        ]

        positive = sorted(
            [
                factor
                for factor in factors
                if factor["shap_value"] > 0
            ],
            key=lambda item: item["shap_value"],
            reverse=True,
        )[:top_k]

        negative = sorted(
            [
                factor
                for factor in factors
                if factor["shap_value"] < 0
            ],
            key=lambda item: item["shap_value"],
        )[:top_k]

        all_factors = sorted(
            factors,
            key=lambda item: abs(
                item["shap_value"]
            ),
            reverse=True,
        )

        record: dict[str, Any] = {
            "payment_id": str(
                feature_frame.iloc[
                    row_index
                ]["payment_id"]
            ),
            "heldout_row_position": int(
                row_index
            ),
            "model_version": str(
                bundle["model_version"]
            ),
            "explanation_version": (
                EXPLANATION_VERSION
            ),
            "generated_at_utc": generated_at,
            "target": str(
                bundle["target"]
            ),
            "heldout_report": (
                HELDOUT_REPORT_PATH.name
            ),
            "feature_source": (
                FEATURE_SOURCE_PATH.name
            ),
            "base_value_raw": float(
                base_value
            ),
            "shap_reconstructed_raw_score": float(
                reconstructed_raw[
                    row_index
                ]
            ),
            "top_positive_factors_json": json.dumps(
                positive,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "top_negative_factors_json": json.dumps(
                negative,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "all_feature_contributions_json": json.dumps(
                all_factors,
                ensure_ascii=False,
                sort_keys=True,
            ),
        }

        if raw_scores is not None:
            record[
                "model_raw_score"
            ] = float(
                raw_scores[row_index]
            )

        if reconstruction_error is not None:
            record[
                "shap_reconstruction_error"
            ] = float(
                reconstruction_error[
                    row_index
                ]
            )

        records.append(
            record
        )

    result = pd.DataFrame.from_records(
        records
    )

    if len(result) != len(
        heldout_ids
    ):
        raise AssertionError(
            "Explanation row count does not match held-out row count"
        )

    output_ids = (
        result["payment_id"]
        .astype(str)
        .tolist()
    )

    if output_ids != heldout_ids:
        raise AssertionError(
            "Explanation rows are not aligned with held-out IDs"
        )

    if result[
        "payment_id"
    ].duplicated().any():
        raise AssertionError(
            "Explanation output contains duplicate payment IDs"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        output_path,
        index=False,
    )

    print(
        "Held-out report:",
        HELDOUT_REPORT_PATH.relative_to(
            ROOT
        ),
    )

    print(
        "Feature source:",
        FEATURE_SOURCE_PATH.relative_to(
            ROOT
        ),
    )

    print(
        "SHAP explanation generation complete"
    )

    print(
        f"Rows: {len(result):,}"
    )

    try:
      display_output_path = output_path.relative_to(ROOT)
    except ValueError:
      display_output_path = output_path

    print(
      "Output:",
      display_output_path,
    )

    print(
        "Model version:",
        bundle["model_version"],
    )

    print(
        "Explanation version:",
        EXPLANATION_VERSION,
    )

    if reconstruction_error is not None:
        print(
            "Max SHAP reconstruction error:",
            f"{reconstruction_error.max():.10f}",
        )

    return result


def main() -> None:
    generate_explanations()


if __name__ == "__main__":
    main()