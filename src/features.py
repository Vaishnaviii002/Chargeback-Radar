import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
ARTIFACT_DIR = Path("artifacts")

TARGET_COLUMN = "chargeback_within_120d"

CATEGORICAL_FEATURES = [
    "card_network",
    "product_category",
    "cvv_result",
    "threeds_status",
]

NUMERICAL_FEATURES = [
    "amount_paise",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_digital_good",
    "descriptor_clarity_score",
    "phone_verified",
    "email_verified",
    "account_age_days",
    "has_prior_order",
    "total_prior_orders",
    "prior_disputes_count",
    "days_since_last_order",
    "txns_last_1h",
    "txns_last_24h",
    "txns_last_7d",
    "amount_last_24h_paise",
    "device_is_new",
    "ip_country_matches_billing",
    "ip_is_proxy_or_vpn",
    "threeds_liability_shift",
    "billing_shipping_distance_km",
]

MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

FORBIDDEN_EXACT_FEATURES = {
    "chargeback_within_120d",
    "chargeback_family",
    "chargeback_reason_code",
    "dispute_created_at",
    "dispute_status",
    "reason_code",
    "final_outcome",
    "was_won",
    "was_lost",
}

def assert_no_forbidden_features() -> None:
    for column in MODEL_FEATURES:
        if column in FORBIDDEN_EXACT_FEATURES:
            raise ValueError(
                f"Future-data leakage detected in feature: {column}"
            )

        if column.startswith(("future_", "final_", "label_")):
            raise ValueError(
                f"Future-data leakage detected in feature: {column}"
            )


def add_basic_time_features(
    payments: pd.DataFrame,
) -> pd.DataFrame:
    payments = payments.copy()

    payments["hour_of_day"] = payments["created_at"].dt.hour
    payments["day_of_week"] = payments["created_at"].dt.dayofweek

    payments["is_weekend"] = (
        payments["day_of_week"] >= 5
    ).astype(int)

    return payments


def add_customer_history_features(
    payments: pd.DataFrame,
) -> pd.DataFrame:
    payments = payments.copy()
    payments = payments.sort_values("created_at").reset_index(drop=True)

    customer_groups = payments.groupby(
        "customer_id",
        sort=False,
    )

    payments["total_prior_orders"] = customer_groups.cumcount()

    payments["has_prior_order"] = (
        payments["total_prior_orders"] > 0
    ).astype(int)

    previous_order_time = customer_groups["created_at"].shift(1)

    payments["days_since_last_order"] = (
        payments["created_at"] - previous_order_time
    ).dt.total_seconds() / 86_400

    payments["days_since_last_order"] = (
        payments["days_since_last_order"]
        .fillna(999.0)
        .clip(lower=0, upper=999)
    )

    return payments


def add_velocity_features(
    payments: pd.DataFrame,
) -> pd.DataFrame:
    payments = payments.copy()
    payments = payments.sort_values("created_at").reset_index(drop=True)

    customer_history: dict[
        str,
        deque[tuple[pd.Timestamp, int]],
    ] = defaultdict(deque)

    txns_last_1h: list[int] = []
    txns_last_24h: list[int] = []
    txns_last_7d: list[int] = []
    amount_last_24h: list[int] = []

    one_hour = pd.Timedelta(hours=1)
    one_day = pd.Timedelta(days=1)
    seven_days = pd.Timedelta(days=7)

    for row in payments.itertuples(index=False):
        history = customer_history[row.customer_id]
        current_time = row.created_at

        while (
            history
            and current_time - history[0][0] > seven_days
        ):
            history.popleft()

        count_1h = 0
        count_24h = 0
        count_7d = len(history)
        amount_24h = 0

        for previous_time, previous_amount in history:
            age = current_time - previous_time

            if age <= one_hour:
                count_1h += 1

            if age <= one_day:
                count_24h += 1
                amount_24h += previous_amount

        txns_last_1h.append(count_1h)
        txns_last_24h.append(count_24h)
        txns_last_7d.append(count_7d)
        amount_last_24h.append(amount_24h)

        history.append(
            (current_time, int(row.amount_paise))
        )

    payments["txns_last_1h"] = txns_last_1h
    payments["txns_last_24h"] = txns_last_24h
    payments["txns_last_7d"] = txns_last_7d
    payments["amount_last_24h_paise"] = amount_last_24h

    return payments


def add_prior_dispute_count(
    payments: pd.DataFrame,
    disputes: pd.DataFrame,
) -> pd.DataFrame:
    payments = payments.copy()

    dispute_customer_map = payments[
        ["payment_id", "customer_id"]
    ]

    disputes_with_customer = disputes.merge(
        dispute_customer_map,
        on="payment_id",
        how="left",
        validate="one_to_one",
    )

    if disputes_with_customer["customer_id"].isna().any():
        raise ValueError(
            "A dispute references an unknown payment."
        )

    dispute_times_by_customer: dict[str, np.ndarray] = {}

    for customer_id, group in disputes_with_customer.groupby(
        "customer_id"
    ):
        dispute_times_by_customer[customer_id] = np.sort(
            group["created_at"].astype("int64").to_numpy()
        )

    payment_times = (
        payments["created_at"]
        .astype("int64")
        .to_numpy()
    )

    prior_dispute_counts: list[int] = []

    for customer_id, payment_time in zip(
        payments["customer_id"],
        payment_times,
    ):
        customer_dispute_times = dispute_times_by_customer.get(
            customer_id
        )

        if customer_dispute_times is None:
            prior_dispute_counts.append(0)
            continue

        count = np.searchsorted(
            customer_dispute_times,
            payment_time,
            side="left",
        )

        prior_dispute_counts.append(int(count))

    payments["prior_disputes_count"] = prior_dispute_counts

    return payments


def validate_features(features: pd.DataFrame) -> None:
    assert_no_forbidden_features()

    missing_columns = [
        column
        for column in MODEL_FEATURES
        if column not in features.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing model features: {missing_columns}"
        )

    if features[MODEL_FEATURES].isna().any().any():
        missing = features[MODEL_FEATURES].isna().sum()
        missing = missing[missing > 0]

        raise ValueError(
            f"Missing feature values found:\n{missing}"
        )

    if not (
        features["feature_asof_ts"]
        <= features["created_at"]
    ).all():
        raise ValueError(
            "A feature was calculated after its payment time."
        )

    if features["payment_id"].duplicated().any():
        raise ValueError("Duplicate payment IDs found.")


def build_features() -> pd.DataFrame:
    payments_path = DATA_DIR / "payments.parquet"
    disputes_path = DATA_DIR / "disputes.parquet"

    if not payments_path.exists() or not disputes_path.exists():
        raise FileNotFoundError(
            "Run the data generators before building features."
        )

    payments = pd.read_parquet(payments_path)
    disputes = pd.read_parquet(disputes_path)

    payments["created_at"] = pd.to_datetime(
        payments["created_at"],
        utc=True,
    )

    disputes["created_at"] = pd.to_datetime(
        disputes["created_at"],
        utc=True,
    )

    features = add_basic_time_features(payments)
    features = add_customer_history_features(features)
    features = add_velocity_features(features)
    features = add_prior_dispute_count(features, disputes)

    # Every generated feature represents information available
    # at or before the payment timestamp.
    features["feature_asof_ts"] = features["created_at"]

    metadata_columns = [
        "payment_id",
        "customer_id",
        "created_at",
        "feature_asof_ts",
        "month_index",
        "is_duplicate_payment",
        "cancelled_subscription_billed",
        TARGET_COLUMN,
    ]

    output_columns = (
        metadata_columns
        + MODEL_FEATURES
    )

    features = features[output_columns]

    validate_features(features)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    features.to_parquet(
        DATA_DIR / "features.parquet",
        index=False,
    )

    feature_schema = {
        "target": TARGET_COLUMN,
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "model_features": MODEL_FEATURES,
        "excluded_rule_fields": [
            "is_duplicate_payment",
            "cancelled_subscription_billed",
        ],
    }

    with open(
        ARTIFACT_DIR / "feature_schema.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            feature_schema,
            file,
            indent=2,
        )

    return features


def print_summary(features: pd.DataFrame) -> None:
    print("\nChargeback Radar — Feature Summary")
    print("-" * 48)
    print(f"Rows:                 {len(features):,}")
    print(f"Model features:       {len(MODEL_FEATURES)}")
    print(f"Categorical features: {len(CATEGORICAL_FEATURES)}")
    print(f"Numerical features:   {len(NUMERICAL_FEATURES)}")
    print(
        "Positive labels:      "
        f"{features[TARGET_COLUMN].sum():,}"
    )
    print(
        "Base rate:            "
        f"{features[TARGET_COLUMN].mean():.3%}"
    )
    print(
        "Customers with prior disputes: "
        f"{(features['prior_disputes_count'] > 0).sum():,}"
    )
    print("\nFiles created successfully:")
    print("  data/features.parquet")
    print("  artifacts/feature_schema.json")


def main() -> None:
    features = build_features()
    print_summary(features)


if __name__ == "__main__":
    main()