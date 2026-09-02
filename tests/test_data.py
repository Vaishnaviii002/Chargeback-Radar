from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    payments_path = DATA_DIR / "payments.parquet"
    disputes_path = DATA_DIR / "disputes.parquet"

    assert payments_path.exists(), "payments.parquet is missing"
    assert disputes_path.exists(), "disputes.parquet is missing"

    payments = pd.read_parquet(payments_path)
    disputes = pd.read_parquet(disputes_path)

    return payments, disputes


def test_payment_volume_and_uniqueness():
    payments, _ = load_data()

    assert len(payments) == 80_000
    assert payments["payment_id"].is_unique
    assert payments["customer_id"].notna().all()


def test_chargeback_base_rate():
    payments, _ = load_data()

    base_rate = payments["chargeback_within_120d"].mean()

    assert 0.006 <= base_rate <= 0.009


def test_all_twelve_months_are_present():
    payments, _ = load_data()

    months = sorted(payments["month_index"].unique().tolist())

    assert months == list(range(1, 13))


def test_public_data_does_not_expose_archetype():
    payments, _ = load_data()

    assert "archetype" not in payments.columns


def test_only_supported_card_networks_exist():
    payments, _ = load_data()

    networks = set(payments["card_network"].unique())

    assert networks == {"Visa", "Mastercard"}


def test_every_dispute_has_a_payment():
    payments, disputes = load_data()

    payment_ids = set(payments["payment_id"])
    disputed_payment_ids = set(disputes["payment_id"])

    assert disputed_payment_ids.issubset(payment_ids)


def test_label_count_matches_dispute_count():
    payments, disputes = load_data()

    labelled_chargebacks = int(
        payments["chargeback_within_120d"].sum()
    )

    assert labelled_chargebacks == len(disputes)


def test_disputes_occur_after_payment():
    _, disputes = load_data()

    assert (
        disputes["created_at"]
        > disputes["payment_created_at"]
    ).all()


def test_disputes_occur_within_120_days():
    _, disputes = load_data()

    lag_days = (
        disputes["created_at"]
        - disputes["payment_created_at"]
    ).dt.total_seconds() / 86_400

    assert lag_days.between(20, 120).all()


def test_respond_by_is_after_dispute_creation():
    _, disputes = load_data()

    assert (
        disputes["respond_by"] > disputes["created_at"]
    ).all()


def test_all_chargeback_families_exist():
    _, disputes = load_data()

    expected_families = {
        "true_fraud",
        "friendly_fraud",
        "merchant_error",
    }

    actual_families = set(
        disputes["chargeback_family"].unique()
    )

    assert actual_families == expected_families


def test_no_missing_critical_values():
    payments, disputes = load_data()

    payment_columns = [
        "payment_id",
        "customer_id",
        "created_at",
        "amount_paise",
        "card_network",
        "chargeback_within_120d",
    ]

    dispute_columns = [
        "dispute_id",
        "payment_id",
        "reason_code",
        "chargeback_family",
        "created_at",
    ]

    assert payments[payment_columns].notna().all().all()
    assert disputes[dispute_columns].notna().all().all()