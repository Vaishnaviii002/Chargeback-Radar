from pathlib import Path

import numpy as np
import pandas as pd


SEED = 43
TARGET_CHARGEBACK_RATE = 0.0075
DATA_DIR = Path("data")

BASE_RISK = {
    "NORMAL": 0.0010,
    "SERIAL_DISPUTER": 0.0350,
    "STOLEN_CARD_USER": 0.0900,
    "CONFUSED_LEGIT": 0.0120,
    "DISSATISFIED": 0.0200,
}


def scale_probabilities(
    raw_probabilities: np.ndarray,
    target_rate: float,
) -> np.ndarray:
    """Scale probabilities until their mean matches the target base rate."""

    lower = 0.0
    upper = 20.0

    for _ in range(60):
        multiplier = (lower + upper) / 2
        scaled = np.clip(
            raw_probabilities * multiplier,
            0,
            0.85,
        )

        if scaled.mean() < target_rate:
            lower = multiplier
        else:
            upper = multiplier

    return np.clip(
        raw_probabilities * ((lower + upper) / 2),
        0,
        0.85,
    )


def get_chargeback_family(row: pd.Series) -> str:
    if (
        row["is_duplicate_payment"]
        or row["cancelled_subscription_billed"]
    ):
        return "merchant_error"

    if row["archetype"] == "STOLEN_CARD_USER":
        return "true_fraud"

    if (
        not row["ip_country_matches_billing"]
        and row["ip_is_proxy_or_vpn"]
    ):
        return "true_fraud"

    return "friendly_fraud"


def get_reason_code(
    row: pd.Series,
    family: str,
    rng: np.random.Generator,
) -> str:
    network = row["card_network"]

    if network == "Visa":
        if row["is_duplicate_payment"]:
            return "12.6"

        if row["cancelled_subscription_billed"]:
            return "13.2"

        if family == "true_fraud":
            return "10.4"

        if row["archetype"] == "DISSATISFIED":
            return "13.3"

        return str(rng.choice(["13.1", "13.3", "13.6"]))

    if row["is_duplicate_payment"]:
        return "4834"

    if row["cancelled_subscription_billed"]:
        return "4841"

    if family == "true_fraud":
        return str(rng.choice(["4837", "4863"]))

    return "4853"


def generate_outcomes() -> None:
    rng = np.random.default_rng(SEED)

    raw_path = DATA_DIR / "raw_payments.parquet"

    if not raw_path.exists():
        raise FileNotFoundError(
            "data/raw_payments.parquet was not found. "
            "Run: python -m src.generate"
        )

    payments = pd.read_parquet(raw_path)
    number_of_payments = len(payments)

    # Conditions that are known when the payment is captured.
    payments["is_duplicate_payment"] = (
        rng.random(number_of_payments) < 0.003
    )

    subscription_mask = (
        payments["product_category"] == "subscription"
    )

    payments["cancelled_subscription_billed"] = (
        subscription_mask
        & (rng.random(number_of_payments) < 0.04)
    )

    base_probability = payments["archetype"].map(BASE_RISK).to_numpy()

    risk_multiplier = np.ones(number_of_payments)

    risk_multiplier *= np.where(
        payments["device_is_new"],
        1.70,
        1.0,
    )

    risk_multiplier *= np.where(
        ~payments["ip_country_matches_billing"],
        2.40,
        1.0,
    )

    risk_multiplier *= np.where(
        payments["ip_is_proxy_or_vpn"],
        2.00,
        1.0,
    )

    risk_multiplier *= np.where(
        payments["cvv_result"] == "no_match",
        2.10,
        1.0,
    )

    risk_multiplier *= np.where(
        payments["threeds_status"] == "not_authenticated",
        1.80,
        1.0,
    )

    risk_multiplier *= np.where(
        payments["is_digital_good"],
        1.35,
        1.0,
    )

    risk_multiplier *= np.where(
        payments["amount_paise"] > 1_000_000,
        1.30,
        1.0,
    )

    risk_multiplier *= np.where(
        payments["descriptor_clarity_score"] < 0.35,
        1.60,
        1.0,
    )

    # Successful 3DS authentication reduces fraud liability.
    risk_multiplier *= np.where(
        payments["threeds_liability_shift"],
        0.50,
        1.0,
    )

    raw_probability = base_probability * risk_multiplier

    # Merchant-error conditions receive additional risk.
    raw_probability = np.where(
        payments["is_duplicate_payment"],
        np.maximum(raw_probability, 0.25),
        raw_probability,
    )

    raw_probability = np.where(
        payments["cancelled_subscription_billed"],
        np.maximum(raw_probability, 0.22),
        raw_probability,
    )

    final_probability = scale_probabilities(
        raw_probability,
        TARGET_CHARGEBACK_RATE,
    )

    labels = (
        rng.random(number_of_payments) < final_probability
    )

    payments["chargeback_within_120d"] = labels.astype(int)

    positive_payments = payments.loc[labels].copy()

    families: list[str] = []
    reason_codes: list[str] = []

    for _, row in positive_payments.iterrows():
        family = get_chargeback_family(row)
        reason = get_reason_code(row, family, rng)

        families.append(family)
        reason_codes.append(reason)

    dispute_lag_days = np.clip(
        rng.normal(
            loc=55,
            scale=22,
            size=len(positive_payments),
        ),
        20,
        120,
    ).astype(int)

    dispute_created_at = (
        positive_payments["created_at"].reset_index(drop=True)
        + pd.to_timedelta(dispute_lag_days, unit="D")
    )

    dispute_statuses = rng.choice(
        ["open", "under_review", "won", "lost", "closed"],
        size=len(positive_payments),
        p=[0.20, 0.20, 0.23, 0.32, 0.05],
    )

    disputes = pd.DataFrame(
        {
            "dispute_id": [
                f"disp_demo_{index:010d}"
                for index in range(1, len(positive_payments) + 1)
            ],
            "entity": "dispute",
            "payment_id": positive_payments[
                "payment_id"
            ].reset_index(drop=True),
            "amount": positive_payments[
                "amount_paise"
            ].reset_index(drop=True),
            "amount_deducted": positive_payments[
                "amount_paise"
            ].reset_index(drop=True),
            "currency": "INR",
            "reason_code": reason_codes,
            "chargeback_family": families,
            "respond_by": (
                dispute_created_at + pd.to_timedelta(7, unit="D")
            ),
            "status": dispute_statuses,
            "phase": "chargeback",
            "created_at": dispute_created_at,
            "payment_created_at": positive_payments[
                "created_at"
            ].reset_index(drop=True),
        }
    )

    public_payments = payments.drop(columns=["archetype"])

    realised_rate = public_payments[
        "chargeback_within_120d"
    ].mean()

    if not 0.006 <= realised_rate <= 0.009:
        raise ValueError(
            f"Chargeback rate {realised_rate:.4%} is outside "
            "the required 0.6%-0.9% range."
        )

    if not (
        disputes["created_at"] > disputes["payment_created_at"]
    ).all():
        raise ValueError(
            "A dispute was created before its payment."
        )

    public_payments.to_parquet(
        DATA_DIR / "payments.parquet",
        index=False,
    )

    disputes.to_parquet(
        DATA_DIR / "disputes.parquet",
        index=False,
    )

    print("\nChargeback Radar — Outcome Summary")
    print("-" * 48)
    print(f"Payments:           {len(public_payments):,}")
    print(f"Chargebacks:        {len(disputes):,}")
    print(f"Chargeback rate:    {realised_rate:.3%}")
    print(
        "Earliest dispute:  "
        f"{disputes['created_at'].min()}"
    )
    print(
        "Latest dispute:    "
        f"{disputes['created_at'].max()}"
    )

    print("\nChargeback families:")
    print(
        disputes["chargeback_family"]
        .value_counts()
        .to_string()
    )

    print("\nReason codes:")
    print(
        disputes["reason_code"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nFiles created successfully:")
    print("  data/payments.parquet")
    print("  data/disputes.parquet")


if __name__ == "__main__":
    generate_outcomes()