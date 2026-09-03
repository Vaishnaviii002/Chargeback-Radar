from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
N_CUSTOMERS = 20_000
N_PAYMENTS = 80_000

START_DATE = pd.Timestamp("2025-01-01", tz="UTC")
END_DATE = pd.Timestamp("2026-01-01", tz="UTC")

COHORT_WINDOWS = {
    "train": (
        pd.Timestamp("2025-01-01", tz="UTC"),
        pd.Timestamp("2025-10-01", tz="UTC"),
    ),
    "calibration": (
        pd.Timestamp("2025-10-01", tz="UTC"),
        pd.Timestamp("2025-11-01", tz="UTC"),
    ),
    "test": (
        pd.Timestamp("2025-11-01", tz="UTC"),
        pd.Timestamp("2026-01-01", tz="UTC"),
    ),
}

COHORT_SHARES = {
    "train": 0.75,
    "calibration": 0.10,
    "test": 0.15,
}

DATA_DIR = Path("data")

ARCHETYPE_SHARES = {
    "NORMAL": 0.82,
    "SERIAL_DISPUTER": 0.03,
    "STOLEN_CARD_USER": 0.02,
    "CONFUSED_LEGIT": 0.08,
    "DISSATISFIED": 0.05,
}

NEW_DEVICE_PROBABILITY = {
    "NORMAL": 0.08,
    "SERIAL_DISPUTER": 0.20,
    "STOLEN_CARD_USER": 0.85,
    "CONFUSED_LEGIT": 0.15,
    "DISSATISFIED": 0.12,
}

IP_MISMATCH_PROBABILITY = {
    "NORMAL": 0.02,
    "SERIAL_DISPUTER": 0.08,
    "STOLEN_CARD_USER": 0.65,
    "CONFUSED_LEGIT": 0.04,
    "DISSATISFIED": 0.05,
}

PROXY_PROBABILITY = {
    "NORMAL": 0.01,
    "SERIAL_DISPUTER": 0.08,
    "STOLEN_CARD_USER": 0.55,
    "CONFUSED_LEGIT": 0.02,
    "DISSATISFIED": 0.03,
}

CVV_FAILURE_PROBABILITY = {
    "NORMAL": 0.01,
    "SERIAL_DISPUTER": 0.04,
    "STOLEN_CARD_USER": 0.18,
    "CONFUSED_LEGIT": 0.02,
    "DISSATISFIED": 0.02,
}

THREEDS_SUCCESS_PROBABILITY = {
    "NORMAL": 0.70,
    "SERIAL_DISPUTER": 0.48,
    "STOLEN_CARD_USER": 0.14,
    "CONFUSED_LEGIT": 0.62,
    "DISSATISFIED": 0.58,
}


def mapped_probability(
    archetypes: np.ndarray,
    mapping: dict[str, float],
) -> np.ndarray:
    return np.array([mapping[value] for value in archetypes])


def generate_customers(rng: np.random.Generator) -> pd.DataFrame:
    archetype_names = list(ARCHETYPE_SHARES.keys())
    archetype_probabilities = list(ARCHETYPE_SHARES.values())

    archetypes = rng.choice(
        archetype_names,
        size=N_CUSTOMERS,
        p=archetype_probabilities,
    )

    evaluation_cohorts = rng.choice(
        list(COHORT_SHARES),
        size=N_CUSTOMERS,
        p=list(COHORT_SHARES.values()),
    )

    account_age_at_start = np.zeros(N_CUSTOMERS, dtype=int)

    for archetype in archetype_names:
        mask = archetypes == archetype

        if archetype == "STOLEN_CARD_USER":
            ages = rng.integers(0, 45, size=mask.sum())
        elif archetype == "SERIAL_DISPUTER":
            ages = rng.integers(30, 900, size=mask.sum())
        else:
            ages = rng.integers(0, 1_500, size=mask.sum())

        account_age_at_start[mask] = ages

    account_created_at = START_DATE - pd.to_timedelta(
        account_age_at_start,
        unit="D",
    )

    customers = pd.DataFrame(
        {
            "customer_id": [
                f"cust_demo_{index:08d}"
                for index in range(N_CUSTOMERS)
            ],
            "archetype": archetypes,
            "evaluation_cohort": evaluation_cohorts,
            "account_created_at": account_created_at,
            "phone_verified": rng.random(N_CUSTOMERS) > 0.05,
            "email_verified": rng.random(N_CUSTOMERS) > 0.03,
        }
    )

    return customers


def generate_payments(
    customers: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    customer_activity = rng.lognormal(
        mean=0,
        sigma=0.9,
        size=N_CUSTOMERS,
    )

    risky_customer_mask = customers["archetype"].isin(
        ["SERIAL_DISPUTER", "STOLEN_CARD_USER"]
    )

    customer_activity[risky_customer_mask] *= 1.8
    customer_activity /= customer_activity.sum()

    customer_indexes = rng.choice(
        N_CUSTOMERS,
        size=N_PAYMENTS,
        p=customer_activity,
    )

    selected_customers = customers.iloc[customer_indexes].reset_index(
        drop=True
    )

    evaluation_cohorts = selected_customers[
        "evaluation_cohort"
    ].to_numpy()

    created_at = pd.Series(
        pd.NaT,
        index=range(N_PAYMENTS),
        dtype="datetime64[ns, UTC]",
    )

    for cohort, (window_start, window_end) in (
        COHORT_WINDOWS.items()
    ):
        cohort_mask = evaluation_cohorts == cohort
        window_seconds = int(
            (window_end - window_start).total_seconds()
        )
        random_seconds = rng.integers(
            0,
            window_seconds,
            size=int(cohort_mask.sum()),
        )
        created_at.loc[cohort_mask] = (
            window_start
            + pd.to_timedelta(random_seconds, unit="s")
        )

    if created_at.isna().any():
        raise ValueError("Some payments were not assigned a timestamp.")

    archetypes = selected_customers["archetype"].to_numpy()

    new_device_probability = mapped_probability(
        archetypes,
        NEW_DEVICE_PROBABILITY,
    )

    ip_mismatch_probability = mapped_probability(
        archetypes,
        IP_MISMATCH_PROBABILITY,
    )

    proxy_probability = mapped_probability(
        archetypes,
        PROXY_PROBABILITY,
    )

    cvv_failure_probability = mapped_probability(
        archetypes,
        CVV_FAILURE_PROBABILITY,
    )

    threeds_success_probability = mapped_probability(
        archetypes,
        THREEDS_SUCCESS_PROBABILITY,
    )

    device_is_new = rng.random(N_PAYMENTS) < new_device_probability

    ip_country_matches_billing = ~(
        rng.random(N_PAYMENTS) < ip_mismatch_probability
    )

    ip_is_proxy_or_vpn = rng.random(N_PAYMENTS) < proxy_probability

    cvv_failed = rng.random(N_PAYMENTS) < cvv_failure_probability
    cvv_not_provided = (~cvv_failed) & (
        rng.random(N_PAYMENTS) < 0.04
    )

    cvv_result = np.where(
        cvv_failed,
        "no_match",
        np.where(cvv_not_provided, "not_provided", "match"),
    )

    threeds_success = (
        rng.random(N_PAYMENTS) < threeds_success_probability
    )

    threeds_attempted = (~threeds_success) & (
        rng.random(N_PAYMENTS) < 0.45
    )

    threeds_status = np.where(
        threeds_success,
        "authenticated",
        np.where(threeds_attempted, "attempted", "not_authenticated"),
    )

    categories = rng.choice(
        [
            "fashion",
            "electronics",
            "travel",
            "food",
            "subscription",
            "gaming",
            "education",
        ],
        size=N_PAYMENTS,
        p=[0.20, 0.16, 0.10, 0.15, 0.13, 0.12, 0.14],
    )

    risky_digital_mask = np.isin(
        archetypes,
        ["SERIAL_DISPUTER", "STOLEN_CARD_USER"],
    ) & (rng.random(N_PAYMENTS) < 0.35)

    categories[risky_digital_mask] = rng.choice(
        ["gaming", "subscription", "education"],
        size=risky_digital_mask.sum(),
    )

    is_digital_good = np.isin(
        categories,
        ["gaming", "subscription", "education"],
    )

    median_amounts = np.select(
        [
            categories == "electronics",
            categories == "travel",
            categories == "fashion",
            categories == "subscription",
        ],
        [7_500, 9_000, 2_200, 999],
        default=1_500,
    )

    amount_rupees = rng.lognormal(
        mean=np.log(median_amounts),
        sigma=0.65,
    )

    amount_rupees = np.clip(
        amount_rupees,
        100,
        100_000,
    )

    descriptor_alpha = np.where(
        archetypes == "CONFUSED_LEGIT",
        2.5,
        8.0,
    )

    descriptor_beta = np.where(
        archetypes == "CONFUSED_LEGIT",
        5.5,
        2.0,
    )

    descriptor_clarity_score = rng.beta(
        descriptor_alpha,
        descriptor_beta,
    )

    payments = pd.DataFrame(
        {
            "customer_id": selected_customers["customer_id"],
            "archetype": archetypes,
            "evaluation_cohort": evaluation_cohorts,
            "account_created_at": selected_customers[
                "account_created_at"
            ],
            "phone_verified": selected_customers["phone_verified"],
            "email_verified": selected_customers["email_verified"],
            "created_at": created_at,
            "amount_paise": np.round(amount_rupees * 100).astype(int),
            "currency": "INR",
            "method": "card",
            "card_network": rng.choice(
                ["Visa", "Mastercard"],
size=N_PAYMENTS,
p=[0.56, 0.44],
            ),
            "product_category": categories,
            "is_digital_good": is_digital_good,
            "descriptor_clarity_score": np.round(
                descriptor_clarity_score,
                4,
            ),
            "device_is_new": device_is_new,
            "ip_country_matches_billing": (
                ip_country_matches_billing
            ),
            "ip_is_proxy_or_vpn": ip_is_proxy_or_vpn,
            "cvv_result": cvv_result,
            "threeds_status": threeds_status,
            "threeds_liability_shift": threeds_success,
            "billing_shipping_distance_km": np.round(
                rng.lognormal(3.2, 1.1, N_PAYMENTS),
                2,
            ),
        }
    )

    payments = payments.sort_values("created_at").reset_index(drop=True)

    payments.insert(
        0,
        "payment_id",
        [
            f"pay_demo_{index:012d}"
            for index in range(1, len(payments) + 1)
        ],
    )

    payments["account_age_days"] = (
        payments["created_at"] - payments["account_created_at"]
    ).dt.days

    payments["month_index"] = (
        payments["created_at"].dt.month
    )

    return payments


def print_summary(
    customers: pd.DataFrame,
    payments: pd.DataFrame,
) -> None:
    print("\nChargeback Radar — Raw Dataset Summary")
    print("-" * 48)
    print(f"Customers: {len(customers):,}")
    print(f"Payments:  {len(payments):,}")
    print(
        f"Period:    {payments['created_at'].min()} "
        f"to {payments['created_at'].max()}"
    )

    print("\nCustomer archetypes:")
    print(
        customers["archetype"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .astype(str)
        .add("%")
    )

    print("\nPayments per month:")
    print(
        payments.groupby("month_index")
        .size()
        .rename("payments")
    )


def main() -> None:
    rng = np.random.default_rng(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    customers = generate_customers(rng)
    payments = generate_payments(customers, rng)

    customers.to_parquet(
        DATA_DIR / "customers.parquet",
        index=False,
    )

    payments.to_parquet(
        DATA_DIR / "raw_payments.parquet",
        index=False,
    )

    print_summary(customers, payments)
    print("\nFiles created successfully:")
    print("  data/customers.parquet")
    print("  data/raw_payments.parquet")


if __name__ == "__main__":
    main()
