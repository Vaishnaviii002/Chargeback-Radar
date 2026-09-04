from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Final

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import OneHotEncoder

from src.calibrate import (
    fit_isotonic,
    fit_sigmoid,
    predict_sigmoid,
)
from src.evaluate import (
    calculate_ece,
    calculate_precision_at_fraction,
)
from src.train import (
    SEED,
    make_time_splits,
    validate_split_labels,
)


ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = (
    ROOT
    / "data"
    / "features.parquet"
)

FEATURE_SCHEMA_PATH = (
    ROOT
    / "artifacts"
    / "feature_schema.json"
)

SUPPORT_SIGNALS_PATH = (
    ROOT
    / "reports"
    / "support_signals.parquet"
)

CANONICAL_METRICS_PATH = (
    ROOT
    / "reports"
    / "metrics.json"
)

CANONICAL_CALIBRATION_PATH = (
    ROOT
    / "reports"
    / "calibration_metrics.json"
)

ABLATION_REPORT_PATH = (
    ROOT
    / "reports"
    / "ablation.json"
)

ABLATION_PREDICTIONS_PATH = (
    ROOT
    / "reports"
    / "ablation_test_predictions.parquet"
)


REPORT_VERSION: Final = "ablation-v1"

TARGET: Final = (
    "chargeback_within_120d"
)

REVIEW_CAPACITY: Final = 0.02

SIGNAL_FEATURES: Final = [
    "intent_to_cancel",
    "non_receipt_complaint",
    "dissatisfaction",
]


FORBIDDEN_SUPPORT_COLUMNS: Final = {
    "customer_id",
    "customer_name",
    "customer_email",
    "customer_phone",
    "archetype",
    "message_text",
    "support_event_id",
    "chargeback_within_120d",
    "chargeback_outcome",
    "true_fraud",
    "dispute_id",
    "dispute_status",
    "reason_code",
    "chargeback_family",
    "respond_by",
}


DELTA_METRICS: Final = [
    "average_precision",
    "pr_auc",
    "precision",
    "recall",
    "f1",
    "brier_score",
    "expected_calibration_error",
    "precision_at_1_percent",
    "precision_at_5_percent",
    "flagged_count",
    "flagged_rate",
    "true_positive",
    "false_positive",
    "false_negative",
    "true_negative",
]


def _load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required JSON file is missing: {path}"
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


def _model_parameters(
    *,
    scale_pos_weight: float,
) -> dict[str, Any]:
    # This exactly mirrors src/train.py.
    return {
        "objective": "binary",
        "n_estimators": 800,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_child_samples": 100,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 0.2,
        "scale_pos_weight": (
            scale_pos_weight
        ),
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": -1,
    }


def _merge_support_signals(
    features: pd.DataFrame,
    signals: pd.DataFrame,
) -> pd.DataFrame:
    required_signal_columns = {
        "payment_id",
        "scoring_at",
        *SIGNAL_FEATURES,
    }

    missing = sorted(
        required_signal_columns
        - set(signals.columns)
    )

    if missing:
        raise ValueError(
            "Support-signal report is missing "
            "required columns: "
            + ", ".join(missing)
        )

    leaked_columns = sorted(
        set(signals.columns)
        & FORBIDDEN_SUPPORT_COLUMNS
    )

    if leaked_columns:
        raise ValueError(
            "Forbidden fields found in support-"
            "signal report: "
            + ", ".join(leaked_columns)
        )

    if features[
        "payment_id"
    ].duplicated().any():
        raise ValueError(
            "Feature table contains duplicate "
            "payment IDs"
        )

    if signals[
        "payment_id"
    ].isna().any():
        raise ValueError(
            "Support-signal report contains "
            "missing payment IDs"
        )

    signals = signals.copy()

    signals["payment_id"] = (
        signals["payment_id"]
        .astype(str)
    )

    if signals[
        "payment_id"
    ].duplicated().any():
        raise ValueError(
            "Support-signal report contains "
            "duplicate payment IDs"
        )

    feature_ids = (
        features["payment_id"]
        .astype(str)
        .tolist()
    )

    signal_ids = (
        signals["payment_id"]
        .astype(str)
        .tolist()
    )

    if set(feature_ids) != set(
        signal_ids
    ):
        missing_ids = (
            set(feature_ids)
            - set(signal_ids)
        )

        extra_ids = (
            set(signal_ids)
            - set(feature_ids)
        )

        raise ValueError(
            "Support-signal payment alignment "
            "failed: "
            f"missing={len(missing_ids)}, "
            f"extra={len(extra_ids)}"
        )

    for column in SIGNAL_FEATURES:
        if signals[column].isna().any():
            raise ValueError(
                f"Support signal {column} "
                "contains missing values"
            )

        if str(
            signals[column].dtype
        ) not in {
            "bool",
            "boolean",
        }:
            raise ValueError(
                f"Support signal {column} "
                "must have boolean dtype"
            )

    safe_signals = signals[
        [
            "payment_id",
            "scoring_at",
            *SIGNAL_FEATURES,
        ]
    ].copy()

    merged = features.copy()

    merged["payment_id"] = (
        merged["payment_id"]
        .astype(str)
    )

    merged = merged.merge(
        safe_signals,
        on="payment_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if (
        merged["payment_id"].tolist()
        != feature_ids
    ):
        raise AssertionError(
            "Support merge changed feature "
            "row ordering"
        )

    created_at = pd.to_datetime(
        merged["created_at"],
        utc=True,
        errors="coerce",
    )

    scoring_at = pd.to_datetime(
        merged["scoring_at"],
        utc=True,
        errors="coerce",
    )

    if (
        created_at.isna().any()
        or scoring_at.isna().any()
    ):
        raise ValueError(
            "Invalid created_at or scoring_at "
            "timestamps"
        )

    expected_scoring_at = (
        created_at
        + pd.Timedelta(
            days=7
        )
    )

    if not (
        scoring_at
        == expected_scoring_at
    ).all():
        raise ValueError(
            "Support scoring timestamps do not "
            "match the declared payment-plus-7d "
            "policy"
        )

    merged["scoring_at"] = (
        scoring_at
    )

    return merged


def _make_preprocessor(
    *,
    categorical_features: list[str],
    numerical_features: list[str],
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_features,
            ),
            (
                "numerical",
                "passthrough",
                numerical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _select_calibrator(
    *,
    calibration_created_at: pd.Series,
    calibration_labels: np.ndarray,
    calibration_raw_probability: np.ndarray,
) -> tuple[
    str,
    Any,
    dict[str, float],
]:
    calibration_frame = pd.DataFrame(
        {
            "created_at": (
                calibration_created_at
                .reset_index(drop=True)
            ),
            "label": (
                calibration_labels
            ),
            "raw_probability": (
                calibration_raw_probability
            ),
        }
    )

    calibration_frame = (
        calibration_frame
        .sort_values(
            "created_at"
        )
        .reset_index(drop=True)
    )

    split_index = int(
        len(calibration_frame)
        * 0.70
    )

    calibration_fit = (
        calibration_frame.iloc[
            :split_index
        ]
    )

    calibration_selection = (
        calibration_frame.iloc[
            split_index:
        ]
    )

    for name, frame in (
        (
            "Calibration fit",
            calibration_fit,
        ),
        (
            "Calibration selection",
            calibration_selection,
        ),
    ):
        if frame[
            "label"
        ].nunique() != 2:
            raise ValueError(
                f"{name} does not contain "
                "both classes"
            )

    fit_probability = (
        calibration_fit[
            "raw_probability"
        ].to_numpy()
    )

    fit_labels = (
        calibration_fit[
            "label"
        ].to_numpy()
    )

    selection_probability = (
        calibration_selection[
            "raw_probability"
        ].to_numpy()
    )

    selection_labels = (
        calibration_selection[
            "label"
        ].to_numpy()
    )

    sigmoid_candidate = fit_sigmoid(
        fit_probability,
        fit_labels,
    )

    isotonic_candidate = fit_isotonic(
        fit_probability,
        fit_labels,
    )

    sigmoid_selection_probability = (
        predict_sigmoid(
            sigmoid_candidate,
            selection_probability,
        )
    )

    isotonic_selection_probability = (
        isotonic_candidate.predict(
            selection_probability
        )
    )

    sigmoid_selection_brier = (
        brier_score_loss(
            selection_labels,
            sigmoid_selection_probability,
        )
    )

    isotonic_selection_brier = (
        brier_score_loss(
            selection_labels,
            isotonic_selection_probability,
        )
    )

    if (
        isotonic_selection_brier
        < sigmoid_selection_brier
    ):
        selected_method = "isotonic"
    else:
        selected_method = "sigmoid"

    if selected_method == "isotonic":
        final_calibrator = fit_isotonic(
            calibration_raw_probability,
            calibration_labels,
        )
    else:
        final_calibrator = fit_sigmoid(
            calibration_raw_probability,
            calibration_labels,
        )

    selection_metrics = {
        "sigmoid_selection_brier": float(
            sigmoid_selection_brier
        ),
        "isotonic_selection_brier": float(
            isotonic_selection_brier
        ),
    }

    return (
        selected_method,
        final_calibrator,
        selection_metrics,
    )


def _apply_selected_calibrator(
    *,
    method: str,
    calibrator: Any,
    probability: np.ndarray,
) -> np.ndarray:
    if method == "isotonic":
        calibrated = calibrator.predict(
            probability
        )
    elif method == "sigmoid":
        calibrated = predict_sigmoid(
            calibrator,
            probability,
        )
    else:
        raise ValueError(
            "Unsupported calibration method: "
            f"{method}"
        )

    return np.clip(
        calibrated,
        1e-6,
        1 - 1e-6,
    )


def _evaluate_probabilities(
    *,
    calibration_probability: np.ndarray,
    test_labels: np.ndarray,
    test_probability: np.ndarray,
) -> dict[str, float | int]:
    operating_threshold = float(
        np.quantile(
            calibration_probability,
            1 - REVIEW_CAPACITY,
        )
    )

    predicted_labels = (
        test_probability
        >= operating_threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        test_labels,
        predicted_labels,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        test_labels,
        predicted_labels,
        zero_division=0,
    )

    recall = recall_score(
        test_labels,
        predicted_labels,
        zero_division=0,
    )

    f1 = f1_score(
        test_labels,
        predicted_labels,
        zero_division=0,
    )

    average_precision = (
        average_precision_score(
            test_labels,
            test_probability,
        )
    )

    precision_values, recall_values, _ = (
        precision_recall_curve(
            test_labels,
            test_probability,
        )
    )

    pr_auc = auc(
        recall_values[::-1],
        precision_values[::-1],
    )

    brier_score = brier_score_loss(
        test_labels,
        test_probability,
    )

    ece = calculate_ece(
        test_labels,
        test_probability,
    )

    precision_at_1_percent = (
        calculate_precision_at_fraction(
            test_labels,
            test_probability,
            0.01,
        )
    )

    precision_at_5_percent = (
        calculate_precision_at_fraction(
            test_labels,
            test_probability,
            0.05,
        )
    )

    flagged_count = int(
        predicted_labels.sum()
    )

    return {
        "average_precision": float(
            average_precision
        ),
        "pr_auc": float(
            pr_auc
        ),
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1": float(f1),
        "brier_score": float(
            brier_score
        ),
        "expected_calibration_error": float(
            ece
        ),
        "precision_at_1_percent": float(
            precision_at_1_percent
        ),
        "precision_at_5_percent": float(
            precision_at_5_percent
        ),
        "operating_threshold": (
            operating_threshold
        ),
        "flagged_count": (
            flagged_count
        ),
        "flagged_rate": float(
            flagged_count
            / len(test_labels)
        ),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
    }


def _train_variant(
    *,
    name: str,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    model_features: list[str],
    categorical_features: list[str],
    numerical_features: list[str],
    target: str,
) -> dict[str, Any]:
    X_train = train[
        model_features
    ]

    y_train = train[
        target
    ]

    X_calibration = calibration[
        model_features
    ]

    y_calibration = calibration[
        target
    ]

    X_test = test[
        model_features
    ]

    y_test = test[
        target
    ]

    preprocessor = _make_preprocessor(
        categorical_features=(
            categorical_features
        ),
        numerical_features=(
            numerical_features
        ),
    )

    X_train_transformed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_calibration_transformed = (
        preprocessor.transform(
            X_calibration
        )
    )

    X_test_transformed = (
        preprocessor.transform(
            X_test
        )
    )

    positive_count = int(
        y_train.sum()
    )

    negative_count = int(
        len(y_train)
        - positive_count
    )

    scale_pos_weight = (
        negative_count
        / positive_count
    )

    model = lgb.LGBMClassifier(
        **_model_parameters(
            scale_pos_weight=(
                scale_pos_weight
            )
        )
    )

    model.fit(
        X_train_transformed,
        y_train,
        eval_set=[
            (
                X_calibration_transformed,
                y_calibration,
            )
        ],
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50,
                verbose=False,
            )
        ],
    )

    calibration_raw_probability = (
        model.predict_proba(
            X_calibration_transformed
        )[:, 1]
    )

    test_raw_probability = (
        model.predict_proba(
            X_test_transformed
        )[:, 1]
    )

    (
        selected_method,
        calibrator,
        selection_metrics,
    ) = _select_calibrator(
        calibration_created_at=(
            calibration[
                "created_at"
            ]
        ),
        calibration_labels=(
            y_calibration.to_numpy()
        ),
        calibration_raw_probability=(
            calibration_raw_probability
        ),
    )

    calibration_probability = (
        _apply_selected_calibrator(
            method=selected_method,
            calibrator=calibrator,
            probability=(
                calibration_raw_probability
            ),
        )
    )

    test_probability = (
        _apply_selected_calibrator(
            method=selected_method,
            calibrator=calibrator,
            probability=(
                test_raw_probability
            ),
        )
    )

    metrics = _evaluate_probabilities(
        calibration_probability=(
            calibration_probability
        ),
        test_labels=(
            y_test.to_numpy()
        ),
        test_probability=(
            test_probability
        ),
    )

    return {
        "name": name,
        "model_version": (
            f"ablation-{name}-v1"
        ),
        "features": list(
            model_features
        ),
        "feature_count": len(
            model_features
        ),
        "categorical_features": list(
            categorical_features
        ),
        "numerical_features": list(
            numerical_features
        ),
        "seed": SEED,
        "best_iteration": int(
            model.best_iteration_
        ),
        "scale_pos_weight": float(
            scale_pos_weight
        ),
        "selected_calibration_method": (
            selected_method
        ),
        "calibration_selection": (
            selection_metrics
        ),
        "metrics": metrics,
        "_test_probability": (
            test_probability
        ),
    }


def _metric_deltas(
    *,
    baseline: dict[str, Any],
    enhanced: dict[str, Any],
) -> dict[str, float | int]:
    baseline_metrics = (
        baseline["metrics"]
    )

    enhanced_metrics = (
        enhanced["metrics"]
    )

    return {
        metric: (
            enhanced_metrics[metric]
            - baseline_metrics[metric]
        )
        for metric in DELTA_METRICS
    }


def _canonical_comparison(
    baseline: dict[str, Any],
    *,
    metrics_path: Path,
    calibration_path: Path,
) -> dict[str, Any] | None:
    if (
        not metrics_path.exists()
        or not calibration_path.exists()
    ):
        return None

    canonical_metrics = (
        _load_json(
            metrics_path
        )
    )

    canonical_calibration = (
        _load_json(
            calibration_path
        )
    )

    model_performance = (
        canonical_metrics[
            "model_performance"
        ]
    )

    reference = {
        "average_precision": (
            model_performance[
                "average_precision"
            ]
        ),
        "precision": (
            model_performance[
                "precision"
            ]
        ),
        "recall": (
            model_performance[
                "recall"
            ]
        ),
        "f1": (
            model_performance[
                "f1"
            ]
        ),
        "brier_score": (
            canonical_calibration[
                "test_calibrated_brier"
            ]
        ),
        "expected_calibration_error": (
            model_performance[
                "expected_calibration_error"
            ]
        ),
        "precision_at_1_percent": (
            model_performance[
                "precision_at_1_percent"
            ]
        ),
        "precision_at_5_percent": (
            model_performance[
                "precision_at_5_percent"
            ]
        ),
    }

    differences = {
        metric: float(
            baseline["metrics"][metric]
            - value
        )
        for metric, value
        in reference.items()
    }

    return {
        "canonical_selected_method": (
            canonical_calibration[
                "selected_method"
            ]
        ),
        "ablation_selected_method": (
            baseline[
                "selected_calibration_method"
            ]
        ),
        "metric_differences": differences,
        "maximum_absolute_difference": (
            max(
                abs(value)
                for value
                in differences.values()
            )
        ),
    }


def _public_variant(
    variant: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value
        in variant.items()
        if not key.startswith("_")
    }


def run_ablation(
    *,
    features_path: Path = FEATURES_PATH,
    schema_path: Path = FEATURE_SCHEMA_PATH,
    signals_path: Path = SUPPORT_SIGNALS_PATH,
    report_path: Path = ABLATION_REPORT_PATH,
    predictions_path: Path = (
        ABLATION_PREDICTIONS_PATH
    ),
    canonical_metrics_path: Path = (
        CANONICAL_METRICS_PATH
    ),
    canonical_calibration_path: Path = (
        CANONICAL_CALIBRATION_PATH
    ),
) -> dict[str, Any]:
    for required_path in (
        features_path,
        schema_path,
        signals_path,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required ablation input "
                f"is missing: {required_path}"
            )

    features = pd.read_parquet(
        features_path
    )

    signals = pd.read_parquet(
        signals_path
    )

    schema = _load_json(
        schema_path
    )

    if schema["target"] != TARGET:
        raise ValueError(
            "Feature-schema target does not "
            "match the authoritative target"
        )

    baseline_features = list(
        schema["model_features"]
    )

    categorical_features = list(
        schema[
            "categorical_features"
        ]
    )

    baseline_numerical_features = list(
        schema[
            "numerical_features"
        ]
    )

    if any(
        signal in baseline_features
        for signal in SIGNAL_FEATURES
    ):
        raise ValueError(
            "Baseline feature set already "
            "contains support signals"
        )

    merged = _merge_support_signals(
        features,
        signals,
    )

    train, calibration, test = (
        make_time_splits(
            merged
        )
    )

    for name, frame in (
        ("Training", train),
        ("Calibration", calibration),
        ("Test", test),
    ):
        validate_split_labels(
            name,
            frame,
            TARGET,
        )

    enhanced_features = (
        baseline_features
        + SIGNAL_FEATURES
    )

    enhanced_numerical_features = (
        baseline_numerical_features
        + SIGNAL_FEATURES
    )

    baseline = _train_variant(
        name="baseline",
        train=train,
        calibration=calibration,
        test=test,
        model_features=(
            baseline_features
        ),
        categorical_features=(
            categorical_features
        ),
        numerical_features=(
            baseline_numerical_features
        ),
        target=TARGET,
    )

    enhanced = _train_variant(
        name="enhanced",
        train=train,
        calibration=calibration,
        test=test,
        model_features=(
            enhanced_features
        ),
        categorical_features=(
            categorical_features
        ),
        numerical_features=(
            enhanced_numerical_features
        ),
        target=TARGET,
    )

    if (
        baseline["seed"]
        != enhanced["seed"]
        or baseline[
            "scale_pos_weight"
        ]
        != enhanced[
            "scale_pos_weight"
        ]
    ):
        raise AssertionError(
            "Ablation variants did not use "
            "identical training controls"
        )

    deltas = _metric_deltas(
        baseline=baseline,
        enhanced=enhanced,
    )

    canonical_comparison = (
        _canonical_comparison(
            baseline,
            metrics_path=(
                canonical_metrics_path
            ),
            calibration_path=(
                canonical_calibration_path
            ),
        )
    )

    report = {
        "report_version": (
            REPORT_VERSION
        ),
        "experiment": (
            "Baseline tabular model versus "
            "tabular plus three timestamp-safe "
            "support-text signals"
        ),
        "target": TARGET,
        "seed": SEED,
        "synthetic_data": True,
        "dataset": {
            "training_rows": len(train),
            "calibration_rows": (
                len(calibration)
            ),
            "test_rows": len(test),
            "training_positives": int(
                train[TARGET].sum()
            ),
            "calibration_positives": int(
                calibration[TARGET].sum()
            ),
            "test_positives": int(
                test[TARGET].sum()
            ),
            "test_base_rate": float(
                test[TARGET].mean()
            ),
        },
        "controls": {
            "same_payment_rows": True,
            "same_customer_disjoint_splits": True,
            "same_seed": True,
            "same_lightgbm_parameters": True,
            "same_early_stopping_procedure": True,
            "same_calibration_selection": True,
            "same_evaluation_procedure": True,
            "train_period": "months 1-9",
            "calibration_period": "month 10",
            "test_period": "months 11-12",
            "support_scoring_policy": (
                "payment timestamp plus 7 days"
            ),
            "support_events_after_scoring_excluded": True,
        },
        "baseline": _public_variant(
            baseline
        ),
        "enhanced": _public_variant(
            enhanced
        ),
        "delta_enhanced_minus_baseline": (
            deltas
        ),
        "canonical_baseline_comparison": (
            canonical_comparison
        ),
        "interpretation_rule": (
            "Publish all deltas honestly. "
            "A negative delta means the enhanced "
            "model performed worse on that metric."
        ),
        "limitations": [
            (
                "All payments, outcomes and support "
                "messages are synthetic."
            ),
            (
                "The enhanced model uses support "
                "text observable up to seven days "
                "after payment, while baseline "
                "tabular features are capture-time "
                "features."
            ),
            (
                "This experiment demonstrates "
                "controlled methodology and does "
                "not establish real-world "
                "generalization."
            ),
        ],
    }

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    predictions = pd.DataFrame(
        {
            "payment_id": (
                test["payment_id"]
                .astype(str)
                .tolist()
            ),
            TARGET: (
                test[TARGET]
                .astype(int)
                .tolist()
            ),
            "baseline_probability": (
                baseline[
                    "_test_probability"
                ]
            ),
            "enhanced_probability": (
                enhanced[
                    "_test_probability"
                ]
            ),
        }
    )

    predictions.to_parquet(
        predictions_path,
        index=False,
    )

    return report


def main() -> None:
    report = run_ablation()

    baseline = report[
        "baseline"
    ]

    enhanced = report[
        "enhanced"
    ]

    deltas = report[
        "delta_enhanced_minus_baseline"
    ]

    print(
        "Chargeback Radar — Controlled Ablation"
    )
    print("-" * 58)
    print(
        f"Training rows:    "
        f"{report['dataset']['training_rows']:,}"
    )
    print(
        f"Calibration rows: "
        f"{report['dataset']['calibration_rows']:,}"
    )
    print(
        f"Test rows:        "
        f"{report['dataset']['test_rows']:,}"
    )

    print("\nBaseline:")
    print(
        "Average Precision: "
        f"{baseline['metrics']['average_precision']:.6f}"
    )
    print(
        "Precision:         "
        f"{baseline['metrics']['precision']:.3%}"
    )
    print(
        "Recall:            "
        f"{baseline['metrics']['recall']:.3%}"
    )
    print(
        "Brier score:       "
        f"{baseline['metrics']['brier_score']:.6f}"
    )
    print(
        "Calibration:       "
        f"{baseline['selected_calibration_method']}"
    )

    print("\nEnhanced:")
    print(
        "Average Precision: "
        f"{enhanced['metrics']['average_precision']:.6f}"
    )
    print(
        "Precision:         "
        f"{enhanced['metrics']['precision']:.3%}"
    )
    print(
        "Recall:            "
        f"{enhanced['metrics']['recall']:.3%}"
    )
    print(
        "Brier score:       "
        f"{enhanced['metrics']['brier_score']:.6f}"
    )
    print(
        "Calibration:       "
        f"{enhanced['selected_calibration_method']}"
    )

    print("\nEnhanced minus baseline:")
    print(
        "Average Precision: "
        f"{deltas['average_precision']:+.6f}"
    )
    print(
        "Precision:         "
        f"{deltas['precision']:+.3%}"
    )
    print(
        "Recall:            "
        f"{deltas['recall']:+.3%}"
    )
    print(
        "Brier score:       "
        f"{deltas['brier_score']:+.6f}"
    )

    comparison = report[
        "canonical_baseline_comparison"
    ]

    if comparison is not None:
        print(
            "\nCanonical baseline maximum "
            "absolute difference: "
            f"{comparison['maximum_absolute_difference']:.12f}"
        )

    print(
        "\nOutput:",
        ABLATION_REPORT_PATH.relative_to(
            ROOT
        ),
    )
    print(
        "Predictions:",
        ABLATION_PREDICTIONS_PATH.relative_to(
            ROOT
        ),
    )


if __name__ == "__main__":
    main()