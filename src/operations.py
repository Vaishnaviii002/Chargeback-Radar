from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 44
DATA_DIR = Path("data")
PAYMENTS_PATH = DATA_DIR / "payments.parquet"
OPERATIONS_PATH = DATA_DIR / "operations.parquet"

# This date is after the complete 120-day outcome window.
OBSERVATION_END = pd.Timestamp("2026-05-01", tz="UTC")


def empty_utc_series(index: pd.Index) -> pd.Series:
    return pd.Series(
        pd.NaT,
        index=index,
        dtype="datetime64[ns, UTC]",
    )


def generate_operations(
    payments: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    required_columns = {
        "payment_id",
        "created_at",
        "is_digital_good",
    }

    missing_columns = required_columns - set(payments.columns)

    if missing_columns:
        raise ValueError(
            "Payments are missing required columns: "
            f"{sorted(missing_columns)}"
        )

    operations = payments[
        [
            "payment_id",
            "created_at",
            "is_digital_good",
        ]
    ].copy()

    operations["created_at"] = pd.to_datetime(
        operations["created_at"],
        utc=True,
    )

    operations["shipment_expected"] = (
        ~operations["is_digital_good"].astype(bool)
    )

    operations["shipment_promised_at"] = empty_utc_series(
        operations.index
    )
    operations["shipment_delivered_at"] = empty_utc_series(
        operations.index
    )

    shipment_mask = operations["shipment_expected"]
    shipment_count = int(shipment_mask.sum())

    promised_days = rng.choice(
        [2, 3, 5, 7],
        size=shipment_count,
        p=[0.15, 0.45, 0.30, 0.10],
    )

    promised_values = (
        operations.loc[shipment_mask, "created_at"].reset_index(
            drop=True
        )
        + pd.to_timedelta(promised_days, unit="D")
    )

    delivery_offsets = np.rint(
        rng.normal(
            loc=-1.5,
            scale=0.6,
            size=shipment_count,
        )
    ).astype(int)

    late_delivery_mask = rng.random(shipment_count) < 0.065
    delivery_offsets[late_delivery_mask] = rng.integers(
        1,
        11,
        size=int(late_delivery_mask.sum()),
    )

    delivery_offsets = np.maximum(
        delivery_offsets,
        -promised_days,
    )

    delivered_values = pd.Series(
        promised_values
        + pd.to_timedelta(delivery_offsets, unit="D"),
        dtype="datetime64[ns, UTC]",
    )

    never_delivered_mask = rng.random(shipment_count) < 0.015
    delivered_values.loc[never_delivered_mask] = pd.NaT

    operations.loc[
        shipment_mask,
        "shipment_promised_at",
    ] = promised_values.to_numpy()

    operations.loc[
        shipment_mask,
        "shipment_delivered_at",
    ] = delivered_values.to_numpy()

    operations["refund_requested_at"] = empty_utc_series(
        operations.index
    )
    operations["refund_promised_by"] = empty_utc_series(
        operations.index
    )
    operations["refund_processed_at"] = empty_utc_series(
        operations.index
    )

    refund_requested_mask = (
        rng.random(len(operations)) < 0.035
    )
    refund_count = int(refund_requested_mask.sum())

    refund_request_delays = rng.integers(
        2,
        46,
        size=refund_count,
    )

    refund_requested_values = (
        operations.loc[
            refund_requested_mask,
            "created_at",
        ].reset_index(drop=True)
        + pd.to_timedelta(
            refund_request_delays,
            unit="D",
        )
    )

    refund_sla_days = rng.choice(
        [3, 5, 7],
        size=refund_count,
        p=[0.20, 0.60, 0.20],
    )

    refund_promised_values = (
        refund_requested_values
        + pd.to_timedelta(refund_sla_days, unit="D")
    )

    refund_processing_offsets = rng.integers(
        0,
        refund_sla_days + 1,
        size=refund_count,
    )

    late_refund_mask = rng.random(refund_count) < 0.10
    refund_processing_offsets[late_refund_mask] = (
        refund_sla_days[late_refund_mask]
        + rng.integers(
            1,
            15,
            size=int(late_refund_mask.sum()),
        )
    )

    refund_processed_values = pd.Series(
        refund_requested_values
        + pd.to_timedelta(
            refund_processing_offsets,
            unit="D",
        ),
        dtype="datetime64[ns, UTC]",
    )

    refund_never_processed_mask = (
        rng.random(refund_count) < 0.04
    )
    refund_processed_values.loc[
        refund_never_processed_mask
    ] = pd.NaT

    operations.loc[
        refund_requested_mask,
        "refund_requested_at",
    ] = refund_requested_values.to_numpy()

    operations.loc[
        refund_requested_mask,
        "refund_promised_by",
    ] = refund_promised_values.to_numpy()

    operations.loc[
        refund_requested_mask,
        "refund_processed_at",
    ] = refund_processed_values.to_numpy()

    operations["observation_at"] = OBSERVATION_END

    shipment_due = (
        operations["shipment_promised_at"].notna()
        & (
            operations["shipment_promised_at"]
            < operations["observation_at"]
        )
    )

    operations["shipment_sla_breached"] = (
        shipment_due
        & (
            operations["shipment_delivered_at"].isna()
            | (
                operations["shipment_delivered_at"]
                > operations["shipment_promised_at"]
            )
        )
    )

    refund_due = (
        operations["refund_promised_by"].notna()
        & (
            operations["refund_promised_by"]
            < operations["observation_at"]
        )
    )

    operations["promised_refund_not_processed"] = (
        refund_due
        & (
            operations["refund_processed_at"].isna()
            | (
                operations["refund_processed_at"]
                > operations["refund_promised_by"]
            )
        )
    )

    operations = operations.drop(
        columns=["is_digital_good"]
    )

    validate_operations(operations)

    return operations


def validate_operations(operations: pd.DataFrame) -> None:
    if operations["payment_id"].duplicated().any():
        raise ValueError("Duplicate operational payment IDs found.")

    if operations["payment_id"].isna().any():
        raise ValueError("Operational payment IDs cannot be missing.")

    promised_before_payment = (
        operations["shipment_promised_at"].notna()
        & (
            operations["shipment_promised_at"]
            < operations["created_at"]
        )
    )

    if promised_before_payment.any():
        raise ValueError("Shipment promise predates payment.")

    refund_before_payment = (
        operations["refund_requested_at"].notna()
        & (
            operations["refund_requested_at"]
            < operations["created_at"]
        )
    )

    if refund_before_payment.any():
        raise ValueError("Refund request predates payment.")

    if not operations["shipment_sla_breached"].any():
        raise ValueError("No shipment SLA breaches were generated.")

    if not operations["promised_refund_not_processed"].any():
        raise ValueError("No delayed refunds were generated.")

    forbidden_columns = {
        "chargeback_within_120d",
        "chargeback_family",
        "reason_code",
        "dispute_status",
    }

    leaked_columns = forbidden_columns & set(operations.columns)

    if leaked_columns:
        raise ValueError(
            "Outcome leakage in operations data: "
            f"{sorted(leaked_columns)}"
        )


def main() -> None:
    if not PAYMENTS_PATH.exists():
        raise FileNotFoundError(
            "data/payments.parquet is missing. "
            "Run the generators first."
        )

    payments = pd.read_parquet(PAYMENTS_PATH)
    rng = np.random.default_rng(SEED)

    operations = generate_operations(
        payments,
        rng,
    )

    operations.to_parquet(
        OPERATIONS_PATH,
        index=False,
    )

    print("\nChargeback Radar — Operations Summary")
    print("-" * 52)
    print(f"Operational rows:            {len(operations):,}")
    print(
        "Shipment SLA breaches:      "
        f"{int(operations['shipment_sla_breached'].sum()):,}"
    )
    print(
        "Refunds requested:          "
        f"{int(operations['refund_requested_at'].notna().sum()):,}"
    )
    print(
        "Refund SLA breaches:        "
        f"{int(operations['promised_refund_not_processed'].sum()):,}"
    )
    print("\nFile created successfully:")
    print("  data/operations.parquet")


if __name__ == "__main__":
    main()