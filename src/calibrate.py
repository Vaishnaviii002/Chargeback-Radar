import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
)


REPORTS_DIR = Path("reports")
ARTIFACT_DIR = Path("artifacts")

CALIBRATION_PATH = (
    REPORTS_DIR / "calibration_predictions.parquet"
)

TEST_PATH = REPORTS_DIR / "test_predictions.parquet"

CALIBRATOR_PATH = ARTIFACT_DIR / "calibrator.joblib"

TARGET = "chargeback_within_120d"


def probability_to_logit(
    probability: np.ndarray,
) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)

    return np.log(
        clipped / (1 - clipped)
    ).reshape(-1, 1)


def fit_sigmoid(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> LogisticRegression:
    calibrator = LogisticRegression(
        solver="lbfgs",
        max_iter=1_000,
        random_state=42,
    )

    calibrator.fit(
        probability_to_logit(probabilities),
        labels,
    )

    return calibrator


def predict_sigmoid(
    calibrator: LogisticRegression,
    probabilities: np.ndarray,
) -> np.ndarray:
    return calibrator.predict_proba(
        probability_to_logit(probabilities)
    )[:, 1]


def fit_isotonic(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> IsotonicRegression:
    calibrator = IsotonicRegression(
        out_of_bounds="clip",
        y_min=0,
        y_max=1,
    )

    calibrator.fit(probabilities, labels)

    return calibrator


def build_reliability_data(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict]:
    observed_rate, predicted_probability = calibration_curve(
        labels,
        probabilities,
        n_bins=10,
        strategy="quantile",
    )

    return [
        {
            "predicted_probability": float(predicted),
            "observed_rate": float(observed),
        }
        for predicted, observed in zip(
            predicted_probability,
            observed_rate,
        )
    ]


def calibrate_model() -> None:
    if not CALIBRATION_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "Prediction files are missing. "
            "Run: python -m src.train"
        )

    calibration_data = pd.read_parquet(CALIBRATION_PATH)
    test_data = pd.read_parquet(TEST_PATH)

    calibration_data = calibration_data.sort_values(
        "created_at"
    ).reset_index(drop=True)

    # Use the first 70% to compare calibration methods.
    # The final 30% chooses the better method without touching test data.
    split_index = int(len(calibration_data) * 0.70)

    calibration_fit = calibration_data.iloc[:split_index]
    calibration_selection = calibration_data.iloc[split_index:]

    for name, frame in [
        ("Calibration fit", calibration_fit),
        ("Calibration selection", calibration_selection),
    ]:
        if frame[TARGET].nunique() != 2:
            raise ValueError(
                f"{name} does not contain both classes."
            )

    fit_probability = calibration_fit[
        "raw_probability"
    ].to_numpy()

    fit_labels = calibration_fit[TARGET].to_numpy()

    selection_probability = calibration_selection[
        "raw_probability"
    ].to_numpy()

    selection_labels = calibration_selection[TARGET].to_numpy()

    sigmoid_candidate = fit_sigmoid(
        fit_probability,
        fit_labels,
    )

    isotonic_candidate = fit_isotonic(
        fit_probability,
        fit_labels,
    )

    sigmoid_selection_probability = predict_sigmoid(
        sigmoid_candidate,
        selection_probability,
    )

    isotonic_selection_probability = (
        isotonic_candidate.predict(selection_probability)
    )

    sigmoid_selection_brier = brier_score_loss(
        selection_labels,
        sigmoid_selection_probability,
    )

    isotonic_selection_brier = brier_score_loss(
        selection_labels,
        isotonic_selection_probability,
    )

    if isotonic_selection_brier < sigmoid_selection_brier:
        selected_method = "isotonic"
    else:
        selected_method = "sigmoid"

    full_calibration_probability = calibration_data[
        "raw_probability"
    ].to_numpy()

    full_calibration_labels = calibration_data[
        TARGET
    ].to_numpy()

    if selected_method == "isotonic":
        final_calibrator = fit_isotonic(
            full_calibration_probability,
            full_calibration_labels,
        )
    else:
        final_calibrator = fit_sigmoid(
            full_calibration_probability,
            full_calibration_labels,
        )

    test_raw_probability = test_data[
        "raw_probability"
    ].to_numpy()

    test_labels = test_data[TARGET].to_numpy()

    if selected_method == "isotonic":
        test_calibrated_probability = (
            final_calibrator.predict(test_raw_probability)
        )
    else:
        test_calibrated_probability = predict_sigmoid(
            final_calibrator,
            test_raw_probability,
        )

    test_calibrated_probability = np.clip(
        test_calibrated_probability,
        1e-6,
        1 - 1e-6,
    )

    raw_brier = brier_score_loss(
        test_labels,
        test_raw_probability,
    )

    calibrated_brier = brier_score_loss(
        test_labels,
        test_calibrated_probability,
    )

    raw_log_loss = log_loss(
        test_labels,
        np.clip(
            test_raw_probability,
            1e-6,
            1 - 1e-6,
        ),
    )

    calibrated_log_loss = log_loss(
        test_labels,
        test_calibrated_probability,
    )

    raw_average_precision = average_precision_score(
        test_labels,
        test_raw_probability,
    )

    calibrated_average_precision = average_precision_score(
        test_labels,
        test_calibrated_probability,
    )

    test_data["calibrated_probability"] = (
        test_calibrated_probability
    )

    test_data.to_parquet(
        REPORTS_DIR / "test_scored.parquet",
        index=False,
    )

    calibrator_bundle = {
        "method": selected_method,
        "calibrator": final_calibrator,
        "version": "0.1.0",
    }

    joblib.dump(
        calibrator_bundle,
        CALIBRATOR_PATH,
    )

    reliability_data = build_reliability_data(
        test_labels,
        test_calibrated_probability,
    )

    with open(
        REPORTS_DIR / "calibration_curve.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            reliability_data,
            file,
            indent=2,
        )

    metrics = {
        "selected_method": selected_method,
        "sigmoid_selection_brier": (
            sigmoid_selection_brier
        ),
        "isotonic_selection_brier": (
            isotonic_selection_brier
        ),
        "test_raw_brier": raw_brier,
        "test_calibrated_brier": calibrated_brier,
        "test_raw_log_loss": raw_log_loss,
        "test_calibrated_log_loss": calibrated_log_loss,
        "test_raw_average_precision": (
            raw_average_precision
        ),
        "test_calibrated_average_precision": (
            calibrated_average_precision
        ),
    }

    with open(
        REPORTS_DIR / "calibration_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)

    print("\nChargeback Radar — Probability Calibration")
    print("-" * 55)
    print(
        f"Sigmoid selection Brier:  "
        f"{sigmoid_selection_brier:.6f}"
    )
    print(
        f"Isotonic selection Brier: "
        f"{isotonic_selection_brier:.6f}"
    )
    print(f"Selected method:           {selected_method}")

    print("\nUntouched held-out test results:")
    print(f"Raw Brier score:           {raw_brier:.6f}")
    print(
        f"Calibrated Brier score:    "
        f"{calibrated_brier:.6f}"
    )
    print(f"Raw log loss:              {raw_log_loss:.6f}")
    print(
        f"Calibrated log loss:       "
        f"{calibrated_log_loss:.6f}"
    )
    print(
        f"Raw Average Precision:     "
        f"{raw_average_precision:.6f}"
    )
    print(
        f"Calibrated Avg Precision:  "
        f"{calibrated_average_precision:.6f}"
    )

    print("\nFiles created successfully:")
    print("  artifacts/calibrator.joblib")
    print("  reports/test_scored.parquet")
    print("  reports/calibration_curve.json")
    print("  reports/calibration_metrics.json")


if __name__ == "__main__":
    calibrate_model()