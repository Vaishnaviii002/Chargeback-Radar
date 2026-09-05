import json
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder


SEED = 42

DATA_DIR = Path("data")
ARTIFACT_DIR = Path("artifacts")
REPORTS_DIR = Path("reports")

FEATURES_PATH = DATA_DIR / "features.parquet"
SCHEMA_PATH = ARTIFACT_DIR / "feature_schema.json"
MODEL_PATH = ARTIFACT_DIR / "model_bundle.joblib"


def load_schema() -> dict:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def make_time_splits(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = features[features["month_index"] <= 9].copy()
    calibration = features[
        features["month_index"] == 10
    ].copy()
    test = features[features["month_index"] >= 11].copy()

    train_end = train["created_at"].max()
    calibration_start = calibration["created_at"].min()
    calibration_end = calibration["created_at"].max()
    test_start = test["created_at"].min()

    if not train_end < calibration_start:
        raise ValueError("Training and calibration periods overlap.")

    if not calibration_end < test_start:
        raise ValueError("Calibration and test periods overlap.")

    customer_sets = {
        "train": set(train["customer_id"]),
        "calibration": set(calibration["customer_id"]),
        "test": set(test["customer_id"]),
    }

    overlaps = {
        "train_calibration": (
            customer_sets["train"]
            & customer_sets["calibration"]
        ),
        "train_test": (
            customer_sets["train"]
            & customer_sets["test"]
        ),
        "calibration_test": (
            customer_sets["calibration"]
            & customer_sets["test"]
        ),
    }

    overlapping_counts = {
        name: len(customers)
        for name, customers in overlaps.items()
    }

    if any(overlapping_counts.values()):
        raise ValueError(
            "Customer leakage detected across time splits: "
            f"{overlapping_counts}. Regenerate the dataset."
        )

    return train, calibration, test


def validate_split_labels(
    name: str,
    split: pd.DataFrame,
    target: str,
) -> None:
    label_count = split[target].nunique()

    if label_count != 2:
        raise ValueError(
            f"{name} split does not contain both classes."
        )


def train_model() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "features.parquet is missing. "
            "Run: python -m src.features"
        )

    features = pd.read_parquet(FEATURES_PATH)
    schema = load_schema()

    target = schema["target"]
    model_features = schema["model_features"]
    categorical_features = schema["categorical_features"]
    numerical_features = schema["numerical_features"]

    train, calibration, test = make_time_splits(features)

    validate_split_labels("Training", train, target)
    validate_split_labels("Calibration", calibration, target)
    validate_split_labels("Test", test, target)

    X_train = train[model_features]
    y_train = train[target]

    X_calibration = calibration[model_features]
    y_calibration = calibration[target]

    X_test = test[model_features]
    y_test = test[target]

    preprocessor = ColumnTransformer(
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

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_calibration_transformed = preprocessor.transform(
        X_calibration
    )
    X_test_transformed = preprocessor.transform(X_test)

    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)

    scale_pos_weight = negative_count / positive_count

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=800,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.2,
        scale_pos_weight=scale_pos_weight,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(
        X_train_transformed,
        y_train,
        eval_X=X_calibration_transformed,
        eval_y=y_calibration,
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50,
                verbose=False,
            )
        ],
    )

    calibration_probability = model.predict_proba(
        X_calibration_transformed
    )[:, 1]

    test_probability = model.predict_proba(
        X_test_transformed
    )[:, 1]

    calibration_ap = average_precision_score(
        y_calibration,
        calibration_probability,
    )

    test_ap = average_precision_score(
        y_test,
        test_probability,
    )

    calibration_baseline = y_calibration.mean()
    test_baseline = y_test.mean()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    transformed_feature_names = (
        preprocessor.get_feature_names_out().tolist()
    )

    model_bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "model_features": model_features,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "transformed_feature_names": transformed_feature_names,
        "target": target,
        "model_version": "0.1.0",
    }

    joblib.dump(model_bundle, MODEL_PATH)

    calibration_predictions = calibration[
        [
            "payment_id",
            "customer_id",
            "created_at",
            "amount_paise",
            target,
        ]
    ].copy()

    calibration_predictions["raw_probability"] = (
        calibration_probability
    )

    calibration_predictions.to_parquet(
        REPORTS_DIR / "calibration_predictions.parquet",
        index=False,
    )

    test_predictions = test[
        [
            "payment_id",
            "customer_id",
            "created_at",
            "amount_paise",
            "is_duplicate_payment",
            "cancelled_subscription_billed",
            target,
        ]
    ].copy()

    test_predictions["raw_probability"] = test_probability

    test_predictions.to_parquet(
        REPORTS_DIR / "test_predictions.parquet",
        index=False,
    )

    metadata = {
        "model_version": "0.1.0",
        "seed": SEED,
        "best_iteration": int(model.best_iteration_),
        "training_rows": len(train),
        "calibration_rows": len(calibration),
        "test_rows": len(test),
        "training_positives": int(y_train.sum()),
        "calibration_positives": int(y_calibration.sum()),
        "test_positives": int(y_test.sum()),
        "scale_pos_weight": scale_pos_weight,
        "calibration_base_rate": calibration_baseline,
        "calibration_average_precision": calibration_ap,
        "test_base_rate": test_baseline,
        "test_average_precision": test_ap,
        "customer_disjoint": True,
        "training_unique_customers": int(
            train["customer_id"].nunique()
        ),
        "calibration_unique_customers": int(
            calibration["customer_id"].nunique()
        ),
        "test_unique_customers": int(
            test["customer_id"].nunique()
        ),
    }

    with open(
        ARTIFACT_DIR / "model_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    print("\nChargeback Radar — Model Training")
    print("-" * 52)
    print(f"Training rows:          {len(train):,}")
    print(f"Calibration rows:       {len(calibration):,}")
    print(f"Test rows:              {len(test):,}")
    print(f"Training positives:     {int(y_train.sum()):,}")
    print(f"Best iteration:         {model.best_iteration_}")
    print(f"Class weight:           {scale_pos_weight:.2f}")

    print("\nCalibration period:")
    print(f"Base-rate baseline:     {calibration_baseline:.4f}")
    print(f"Average Precision:      {calibration_ap:.4f}")

    print("\nHeld-out test period:")
    print(f"Base-rate baseline:     {test_baseline:.4f}")
    print(f"Average Precision:      {test_ap:.4f}")

    print("\nLeakage controls:")
    print("Time-based split:       verified")
    print("Customer-disjoint:      verified")

    print("\nFiles created successfully:")
    print("  artifacts/model_bundle.joblib")
    print("  artifacts/model_metadata.json")
    print("  reports/calibration_predictions.parquet")
    print("  reports/test_predictions.parquet")


if __name__ == "__main__":
    train_model()
