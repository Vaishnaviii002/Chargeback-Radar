import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from src.razorpay_api import (
    router as razorpay_router,
)


import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.razorpay_normalizer import (
    CaptureTimeScoringPayload,
)
from src.risk_scoring import (
    RiskScoringError,
    score_capture_time_payment,
)

from src.evidence_api import router as evidence_router

from src.post_payment_rules import evaluate_post_payment_rules

from src.decide import (
    CostParams,
    simulate_policy,
)

from src.model_explanation_api import (
    get_model_explanation as get_stored_model_explanation,
    get_model_explanation_delivery_service,
    get_model_explanation_repository,
    router as model_explanation_router,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
ARTIFACTS_DIR = ROOT / "artifacts"

# Match the rest of the backend's local configuration behavior while
# preserving deployment-provided environment variables.
load_dotenv(ROOT / ".env", override=False)

DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def _parse_allowed_origins(
    raw_value: str | None,
) -> list[str]:
    candidates = (
        raw_value.split(",")
        if raw_value and raw_value.strip()
        else list(DEFAULT_ALLOWED_ORIGINS)
    )
    origins: list[str] = []

    for candidate in candidates:
        origin = candidate.strip().rstrip("/")

        if not origin:
            continue

        if origin == "*":
            raise RuntimeError(
                "Wildcard CORS origins are not permitted."
            )

        parsed = urlsplit(origin)

        try:
            parsed.port
        except ValueError as error:
            raise RuntimeError(
                "BACKEND_ALLOWED_ORIGINS contains an "
                "invalid origin."
            ) from error

        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise RuntimeError(
                "BACKEND_ALLOWED_ORIGINS contains an "
                "invalid origin."
            )

        if origin not in origins:
            origins.append(origin)

    if not origins:
        raise RuntimeError(
            "At least one CORS origin is required."
        )

    return origins

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
    allow_origins=_parse_allowed_origins(
        os.getenv("BACKEND_ALLOWED_ORIGINS")
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(evidence_router)
app.include_router(model_explanation_router)
app.include_router(razorpay_router)

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

    is_duplicate_payment: bool = False
    cancelled_subscription_billed: bool = False




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
def load_feature_data() -> pd.DataFrame:
    path = DATA_DIR / "features.parquet"

    if not path.exists():
        raise FileNotFoundError(
            "data/features.parquet is missing."
        )

    return pd.read_parquet(path)

@lru_cache(maxsize=1)
def load_operation_data() -> pd.DataFrame:
    path = DATA_DIR / "operations.parquet"

    if not path.exists():
        raise FileNotFoundError(
            "data/operations.parquet is missing."
        )

    return pd.read_parquet(path)


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


@app.get("/api/policy/frontier")
def get_policy_frontier():
    return read_json_file(
        REPORTS_DIR / "policy_frontier.json"
    )

@app.get("/api/policy/effectiveness")
def get_effectiveness_sensitivity():
    return read_json_file(
        REPORTS_DIR / "effectiveness_sensitivity.json"
    )

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







@app.get(
    "/api/transactions/{payment_id}/explanation"
)
def get_transaction_explanation_compatibility(
    payment_id: str,
):
    return get_stored_model_explanation(
        payment_id=payment_id,
        use_ai=False,
        repository=(
            get_model_explanation_repository()
        ),
        delivery_service=(
            get_model_explanation_delivery_service()
        ),
    )



@app.get(
    "/api/transactions/{payment_id}/lifecycle"
)
def get_transaction_lifecycle(
    payment_id: str,
    as_of: datetime | None = None,
):
    try:
        operations = load_operation_data()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    row = operations[
        operations["payment_id"] == payment_id
    ]

    if row.empty:
        raise HTTPException(
            status_code=404,
            detail="Operational record not found.",
        )

    operation = dataframe_records(row.head(1))[0]
    evaluated_at = as_of or operation["observation_at"]

    rule_result = evaluate_post_payment_rules(
        operation,
        as_of=evaluated_at,
    )

    return {
        "payment_id": payment_id,
        "lifecycle": operation,
        "rule_result": rule_result,
        "requires_human_approval": rule_result[
            "requires_human_approval"
        ],
        "action_executed": False,
        "disclosure": (
            "Lifecycle rules use only events visible at the "
            "requested as_of time. No action is executed."
        ),
    }

@app.post("/api/score")
def score_payment(
    request: PaymentScoreRequest,
):
    try:
        scoring_payload = (
            CaptureTimeScoringPayload.model_validate(
                request.model_dump()
            )
        )

        score = score_capture_time_payment(
            scoring_payload
        )

    except RiskScoringError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Risk scoring is temporarily unavailable."
            ),
        ) from error

    explanation = {
        "available": False,
        "method": "TreeSHAP",
        "top_factors": [],
        "reason": (
            "Use the dedicated model-explanation endpoint "
            "for validated held-out transaction explanations."
        ),
    }

    return {
        **score.model_dump(),
        "explanation": explanation,
    }
