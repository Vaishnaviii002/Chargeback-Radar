import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from src.calibrate import probability_to_logit


REPORTS_DIR = Path("reports")
ARTIFACT_DIR = Path("artifacts")

TARGET = "chargeback_within_120d"

CALIBRATION_PREDICTIONS_PATH = (
    REPORTS_DIR / "calibration_predictions.parquet"
)

TEST_SCORED_PATH = REPORTS_DIR / "test_scored.parquet"

CALIBRATOR_PATH = ARTIFACT_DIR / "calibrator.joblib"

DEFAULT_REVIEW_CAPACITY = 0.02
DEFAULT_MANUAL_REVIEW_COST = 150.0
DEFAULT_CHARGEBACK_FEE = 1_500.0


def apply_calibrator(
    probabilities: np.ndarray,
    calibrator_bundle: dict,
) -> np.ndarray:
    method = calibrator_bundle["method"]
    calibrator = calibrator_bundle["calibrator"]

    if method == "isotonic":
        calibrated = calibrator.predict(probabilities)
    elif method == "sigmoid":
        calibrated = calibrator.predict_proba(
            probability_to_logit(probabilities)
        )[:, 1]
    else:
        raise ValueError(
            f"Unknown calibration method: {method}"
        )

    return np.clip(calibrated, 1e-6, 1 - 1e-6)


def calculate_precision_at_fraction(
    labels: np.ndarray,
    probabilities: np.ndarray,
    fraction: float,
    *,
    identifiers: np.ndarray,
) -> float:
    if not (
        len(labels)
        == len(probabilities)
        == len(identifiers)
    ):
        raise ValueError(
            "Labels, probabilities, and identifiers must align."
        )

    stable_identifiers = np.asarray(
        identifiers,
        dtype=str,
    )

    if len(set(stable_identifiers)) != len(
        stable_identifiers
    ):
        raise ValueError(
            "Precision-at-fraction identifiers must be unique."
        )

    number_to_select = max(
        1,
        math.ceil(len(labels) * fraction),
    )

    # Isotonic calibration intentionally creates probability ties. A
    # BLAKE2b key derived only from the unique payment ID gives those ties a
    # platform-independent, outcome-blind order without making an ordinal
    # payment ID a hidden ranking signal.
    tie_breakers = [
        hashlib.blake2b(
            identifier.encode("utf-8")
        ).digest()
        for identifier in stable_identifiers
    ]
    selected_indexes = sorted(
        range(len(labels)),
        key=lambda index: (
            -float(probabilities[index]),
            tie_breakers[index],
            stable_identifiers[index],
        ),
    )[:number_to_select]

    return float(labels[selected_indexes].mean())


def calculate_ece(
    labels: np.ndarray,
    probabilities: np.ndarray,
    number_of_bins: int = 10,
) -> float:
    frame = pd.DataFrame(
        {
            "label": labels,
            "probability": probabilities,
        }
    )

    frame["bin"] = pd.qcut(
        frame["probability"],
        q=number_of_bins,
        duplicates="drop",
    )

    total_rows = len(frame)
    ece = 0.0

    for _, group in frame.groupby(
        "bin",
        observed=True,
    ):
        bin_weight = len(group) / total_rows
        observed_rate = group["label"].mean()
        predicted_rate = group["probability"].mean()

        ece += bin_weight * abs(
            observed_rate - predicted_rate
        )

    return float(ece)


def make_precision_recall_data(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict]:
    precision_values, recall_values, thresholds = (
        precision_recall_curve(
            labels,
            probabilities,
        )
    )

    maximum_points = 300

    selected_indexes = np.unique(
        np.linspace(
            0,
            len(precision_values) - 1,
            min(maximum_points, len(precision_values)),
            dtype=int,
        )
    )

    curve_data: list[dict] = []

    for index in selected_indexes:
        threshold = (
            float(thresholds[index])
            if index < len(thresholds)
            else None
        )

        curve_data.append(
            {
                "precision": float(precision_values[index]),
                "recall": float(recall_values[index]),
                "threshold": threshold,
            }
        )

    return curve_data


def evaluate_model() -> None:
    required_files = [
        CALIBRATION_PREDICTIONS_PATH,
        TEST_SCORED_PATH,
        CALIBRATOR_PATH,
    ]

    for required_file in required_files:
        if not required_file.exists():
            raise FileNotFoundError(
                f"{required_file} is missing. "
                "Run training and calibration first."
            )

    calibration_data = pd.read_parquet(
        CALIBRATION_PREDICTIONS_PATH
    )

    test_data = pd.read_parquet(TEST_SCORED_PATH)

    calibrator_bundle = joblib.load(CALIBRATOR_PATH)

    calibration_probability = apply_calibrator(
        calibration_data["raw_probability"].to_numpy(),
        calibrator_bundle,
    )

    # The merchant can review the riskiest 2% of payments.
    # This threshold is selected without looking at test labels.
    operating_threshold = float(
        np.quantile(
            calibration_probability,
            1 - DEFAULT_REVIEW_CAPACITY,
        )
    )

    test_labels = test_data[TARGET].to_numpy()
    test_probability = test_data[
        "calibrated_probability"
    ].to_numpy()

    predicted_labels = (
        test_probability >= operating_threshold
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

    average_precision = average_precision_score(
        test_labels,
        test_probability,
    )

    base_rate = float(test_labels.mean())

    precision_at_1_percent = (
        calculate_precision_at_fraction(
            test_labels,
            test_probability,
            0.01,
            identifiers=test_data[
                "payment_id"
            ].to_numpy(),
        )
    )

    precision_at_5_percent = (
        calculate_precision_at_fraction(
            test_labels,
            test_probability,
            0.05,
            identifiers=test_data[
                "payment_id"
            ].to_numpy(),
        )
    )

    ece = calculate_ece(
        test_labels,
        test_probability,
    )

    flagged_count = int(predicted_labels.sum())
    flagged_rate = flagged_count / len(test_data)

    false_positive_cost = (
        int(fp) * DEFAULT_MANUAL_REVIEW_COST
    )

    test_amount_rupees = (
        test_data["amount_paise"].to_numpy() / 100
    )

    missed_chargeback_exposure = float(
        (
            test_amount_rupees[
                (test_labels == 1) & (predicted_labels == 0)
            ]
            + DEFAULT_CHARGEBACK_FEE
        ).sum()
    )

    detected_chargeback_exposure = float(
        (
            test_amount_rupees[
                (test_labels == 1) & (predicted_labels == 1)
            ]
            + DEFAULT_CHARGEBACK_FEE
        ).sum()
    )

    test_data["predicted_chargeback"] = predicted_labels

    test_data.to_parquet(
        REPORTS_DIR / "test_evaluated.parquet",
        index=False,
    )

    precision_recall_data = make_precision_recall_data(
        test_labels,
        test_probability,
    )

    with open(
        REPORTS_DIR / "precision_recall_curve.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "base_rate": base_rate,
                "operating_threshold": operating_threshold,
                "points": precision_recall_data,
            },
            file,
            indent=2,
        )

    metrics = {
        "dataset": {
            "test_rows": len(test_data),
            "positive_labels": int(test_labels.sum()),
            "base_rate": base_rate,
        },
        "operating_policy": {
            "review_capacity": DEFAULT_REVIEW_CAPACITY,
            "operating_threshold": operating_threshold,
            "flagged_count": flagged_count,
            "flagged_rate": flagged_rate,
        },
        "model_performance": {
            "average_precision": float(average_precision),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "precision_at_1_percent": (
                precision_at_1_percent
            ),
            "precision_at_5_percent": (
                precision_at_5_percent
            ),
            "expected_calibration_error": ece,
        },
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "false_positive_cost": {
            "manual_review_cost_per_case": (
                DEFAULT_MANUAL_REVIEW_COST
            ),
            "false_positive_count": int(fp),
            "total_rupees": float(false_positive_cost),
        },
        "exposure": {
            "assumed_chargeback_fee_rupees": (
                DEFAULT_CHARGEBACK_FEE
            ),
            "detected_chargeback_exposure_rupees": (
                detected_chargeback_exposure
            ),
            "missed_chargeback_exposure_rupees": (
                missed_chargeback_exposure
            ),
        },
        "disclosures": {
            "synthetic_data": True,
            "split": (
                "Train months 1-9, calibration month 10, "
                "held-out test months 11-12; customer IDs "
                "are disjoint across all three splits"
            ),
            "money_note": (
                "Exposure and false-positive costs are "
                "backtested estimates using declared assumptions."
            ),
        },
    }

    with open(
        REPORTS_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)

    print("\nChargeback Radar — Held-Out Evaluation")
    print("-" * 55)
    print(f"Test transactions:       {len(test_data):,}")
    print(f"Actual chargebacks:      {int(test_labels.sum()):,}")
    print(f"Base rate:               {base_rate:.3%}")
    print(f"Operating threshold:     {operating_threshold:.6f}")
    print(f"Flagged transactions:    {flagged_count:,}")
    print(f"Flagged rate:            {flagged_rate:.3%}")

    print("\nModel performance:")
    print(f"Average Precision:       {average_precision:.4f}")
    print(f"Precision:               {precision:.3%}")
    print(f"Recall:                  {recall:.3%}")
    print(f"F1:                      {f1:.4f}")
    print(f"Precision at top 1%:     {precision_at_1_percent:.3%}")
    print(f"Precision at top 5%:     {precision_at_5_percent:.3%}")
    print(f"Calibration error:       {ece:.6f}")

    print("\nConfusion matrix:")
    print(f"True positives:          {tp:,}")
    print(f"False positives:         {fp:,}")
    print(f"False negatives:         {fn:,}")
    print(f"True negatives:          {tn:,}")

    print("\nFalse-positive cost:")
    print(
        f"₹{false_positive_cost:,.2f} "
        f"at ₹{DEFAULT_MANUAL_REVIEW_COST:,.0f} per review"
    )

    print("\nFiles created successfully:")
    print("  reports/metrics.json")
    print("  reports/precision_recall_curve.json")
    print("  reports/test_evaluated.parquet")


if __name__ == "__main__":
    evaluate_model()
