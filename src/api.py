import json
from fastapi import HTTPException
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.calibrate import probability_to_logit
from src.decide import (
    ACTIONS,
    CostParams,
    calculate_expected_costs,
    simulate_policy,
)


REPORTS_DIR = Path("reports")
ARTIFACTS_DIR = Path("artifacts")

app = FastAPI(
    title="Chargeback Radar API",
    description=(
        "Defense-only chargeback detection and "
        "financial policy simulation API"
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PolicyParamsRequest(BaseModel):
    chargeback_fee: float = Field(
        default=1_500,
        ge=0,
        le=10_000,
    )

    gross_margin_rate: float = Field(
        default=0.35,
        ge=0,
        le=1,
    )

    evidence_cost: float = Field(
        default=40,
        ge=0,
        le=5_000,
    )

    manual_review_cost: float = Field(
        default=150,
        ge=0,
        le=10_000,
    )

    customer_friction_rate: float = Field(
        default=0.08,
        ge=0,
        le=1,
    )

    evidence_recovery_rate: float = Field(
        default=0.45,
        ge=0,
        le=1,
    )

    review_prevention_rate: float = Field(
        default=0.65,
        ge=0,
        le=1,
    )

    refund_cost_rate: float = Field(
        default=0.35,
        ge=0,
        le=1,
    )

    risk_program_penalty: float = Field(
        default=800,
        ge=0,
        le=20_000,
    )


class PaymentScoreRequest(BaseModel):
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    amount_paise: int = Field(
        default=250_000,
        ge=100,
        le=100_000_000,
    )

    card_network: Literal[
        "Visa",
        "Mastercard",
    ] = "Visa"

    product_category: Literal[
        "fashion",
        "electronics",
        "travel",
        "food",
        "subscription",
        "gaming",
        "education",
    ] = "electronics"

    is_digital_good: bool = False

    descriptor_clarity_score: float = Field(
        default=0.80,
        ge=0,
        le=1,
    )

    phone_verified: bool = True
    email_verified: bool = True

    account_age_days: int = Field(
        default=365,
        ge=0,
        le=10_000,
    )

    has_prior_order: int = Field(
        default=1,
        ge=0,
        le=1,
    )

    total_prior_orders: int = Field(
        default=5,
        ge=0,
        le=100_000,
    )

    prior_disputes_count: int = Field(
        default=0,
        ge=0,
        le=10_000,
    )

    days_since_last_order: float = Field(
        default=30,
        ge=0,
        le=999,
    )

    txns_last_1h: int = Field(default=0, ge=0)
    txns_last_24h: int = Field(default=0, ge=0)
    txns_last_7d: int = Field(default=1, ge=0)

    amount_last_24h_paise: int = Field(
        default=0,
        ge=0,
    )

    device_is_new: bool = False
    ip_country_matches_billing: bool = True
    ip_is_proxy_or_vpn: bool = False

    cvv_result: Literal[
        "match",
        "no_match",
        "not_provided",
    ] = "match"

    threeds_status: Literal[
        "authenticated",
        "attempted",
        "not_authenticated",
    ] = "authenticated"

    threeds_liability_shift: bool = True

    billing_shipping_distance_km: float = Field(
        default=10,
        ge=0,
        le=50_000,
    )


def read_json_file(path: Path) -> dict:
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"{path} is missing. Run the ML pipeline first."
            ),
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@lru_cache(maxsize=1)
def load_test_data() -> pd.DataFrame:
    path = REPORTS_DIR / "test_scored.parquet"

    if not path.exists():
        raise FileNotFoundError(
            "reports/test_scored.parquet is missing."
        )

    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def load_default_policy_data() -> pd.DataFrame:
    path = REPORTS_DIR / "test_policy.parquet"

    if not path.exists():
        raise FileNotFoundError(
            "reports/test_policy.parquet is missing."
        )

    return pd.read_parquet(path)

@lru_cache(maxsize=1)
def load_model_bundle() -> dict:
    path = ARTIFACTS_DIR / "model_bundle.joblib"

    if not path.exists():
        raise FileNotFoundError(
            "artifacts/model_bundle.joblib is missing."
        )

    return joblib.load(path)


@lru_cache(maxsize=1)
def load_calibrator_bundle() -> dict:
    path = ARTIFACTS_DIR / "calibrator.joblib"

    if not path.exists():
        raise FileNotFoundError(
            "artifacts/calibrator.joblib is missing."
        )

    return joblib.load(path)


def calibrate_probability(
    raw_probability: np.ndarray,
    calibrator_bundle: dict,
) -> np.ndarray:
    method = calibrator_bundle["method"]
    calibrator = calibrator_bundle["calibrator"]

    if method == "isotonic":
        result = calibrator.predict(raw_probability)
    elif method == "sigmoid":
        result = calibrator.predict_proba(
            probability_to_logit(raw_probability)
        )[:, 1]
    else:
        raise ValueError(
            f"Unknown calibration method: {method}"
        )

    return np.clip(result, 1e-6, 1 - 1e-6)


def dataframe_records(
    frame: pd.DataFrame,
) -> list[dict]:
    return json.loads(
        frame.to_json(
            orient="records",
            date_format="iso",
        )
    )


@app.get("/")
def root():
    return {
        "name": "Chargeback Radar API",
        "status": "running",
        "documentation": "/docs",
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "chargeback-radar-api",
        "version": "0.1.0",
    }


@app.get("/api/metrics")
def get_metrics():
    return {
        "evaluation": read_json_file(
            REPORTS_DIR / "metrics.json"
        ),
        "calibration": read_json_file(
            REPORTS_DIR / "calibration_metrics.json"
        ),
        "model": read_json_file(
            ARTIFACTS_DIR / "model_metadata.json"
        ),
        "default_policy": read_json_file(
            REPORTS_DIR / "policy_default.json"
        ),
    }


@app.get("/api/curves")
def get_curves():
    return {
        "precision_recall": read_json_file(
            REPORTS_DIR / "precision_recall_curve.json"
        ),
        "calibration": read_json_file(
            REPORTS_DIR / "calibration_curve.json"
        ),
    }


@app.post("/api/simulate")
def simulate(request: PolicyParamsRequest):
    try:
        test_data = load_test_data()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    params = CostParams(**request.model_dump())

    results, _ = simulate_policy(
        test_data,
        params,
    )

    return results


@app.get("/api/transactions")
def get_transactions(
    limit: int = Query(default=50, ge=1, le=200),
    action: Literal[
        "MONITOR",
        "PREPARE_EVIDENCE",
        "MANUAL_REVIEW",
        "RECOMMEND_REFUND",
    ]
    | None = None,
):
    try:
        frame = load_default_policy_data().copy()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    if action is not None:
        frame = frame[
            frame["recommended_action"] == action
        ]

    frame = frame.sort_values(
        "calibrated_probability",
        ascending=False,
    ).head(limit)

    selected_columns = [
        "payment_id",
        "customer_id",
        "created_at",
        "amount_paise",
        "calibrated_probability",
        "recommended_action",
        "is_duplicate_payment",
        "cancelled_subscription_billed",
        "chargeback_within_120d",
    ]

    return {
        "count": len(frame),
        "transactions": dataframe_records(
            frame[selected_columns]
        ),
    }


@app.get("/api/transactions/{payment_id}")
def get_transaction(payment_id: str):
    try:
        frame = load_default_policy_data()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    transaction = frame[
        frame["payment_id"] == payment_id
    ]

    if transaction.empty:
        raise HTTPException(
            status_code=404,
            detail="Transaction not found.",
        )

    return dataframe_records(transaction)[0]

@app.post("/api/score")
def score_payment(request: PaymentScoreRequest):
    try:
        model_bundle = load_model_bundle()
        calibrator_bundle = load_calibrator_bundle()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    request_data = request.model_dump()
    created_at = request_data.pop("created_at")

    day_of_week = created_at.weekday()

    request_data.update(
        {
            "hour_of_day": created_at.hour,
            "day_of_week": day_of_week,
            "is_weekend": int(day_of_week >= 5),
        }
    )

    model_features = model_bundle["model_features"]

    model_input = pd.DataFrame(
        [request_data]
    )[model_features]

    transformed_input = model_bundle[
        "preprocessor"
    ].transform(model_input)

    raw_probability = model_bundle[
        "model"
    ].predict_proba(transformed_input)[:, 1]

    calibrated_probability = calibrate_probability(
        raw_probability,
        calibrator_bundle,
    )

    probability = float(calibrated_probability[0])
    amount_rupees = request.amount_paise / 100

    default_params = CostParams()

    expected_costs = calculate_expected_costs(
        np.array([probability]),
        np.array([amount_rupees]),
        default_params,
    )[0]

    selected_index = int(np.argmin(expected_costs))
    recommended_action = ACTIONS[selected_index]

    if probability < 0.01:
        risk_band = "LOW"
    elif probability < 0.05:
        risk_band = "MEDIUM"
    elif probability < 0.15:
        risk_band = "HIGH"
    else:
        risk_band = "CRITICAL"

    cost_breakdown = {
        action: float(expected_costs[index])
        for index, action in enumerate(ACTIONS)
    }

    return {
        "raw_probability": float(raw_probability[0]),
        "calibrated_probability": probability,
        "risk_percentage": round(probability * 100, 3),
        "risk_band": risk_band,
        "recommended_action": recommended_action,
        "expected_costs_rupees": cost_breakdown,
        "model_version": model_bundle["model_version"],
        "calibration_method": calibrator_bundle["method"],
        "requires_human_approval": (
            recommended_action == "RECOMMEND_REFUND"
        ),
        "disclosure": (
            "Prediction produced by a model trained on "
            "synthetic data. It is a decision-support signal, "
            "not an automatic financial action."
        ),
    }


@app.get("/api/transactions")
def list_transactions(
    limit: int = 25,
    action: str | None = None,
):
    """Return the highest-risk held-out transactions."""

    limit = max(1, min(limit, 200))
    transactions = load_default_policy_data().copy()

    if action:
        action = action.upper()

        if action not in ACTIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action. Choose one of: {', '.join(ACTIONS)}",
            )

        transactions = transactions[
            transactions["recommended_action"] == action
        ]

    transactions = transactions.sort_values(
        "calibrated_probability",
        ascending=False,
    ).head(limit)

    columns = [
        "payment_id",
        "customer_id",
        "created_at",
        "amount_paise",
        "calibrated_probability",
        "recommended_action",
        "target",
    ]

    return {
        "count": len(transactions),
        "items": dataframe_records(transactions[columns]),
    }


@app.get("/api/transactions/{payment_id}")
def get_transaction(payment_id: str):
    """Return one transaction and its policy decision."""

    transactions = load_default_policy_data()
    transaction = transactions[
        transactions["payment_id"] == payment_id
    ]

    if transaction.empty:
        raise HTTPException(
            status_code=404,
            detail="Transaction not found",
        )

    return dataframe_records(transaction)[0]