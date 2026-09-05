from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from src.calibrate import probability_to_logit
from src.decide import (
    ACTIONS,
    CostParams,
    calculate_expected_costs,
)
from src.razorpay_normalizer import (
    CaptureTimeScoringPayload,
)
from src.rules import evaluate_rules


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "artifacts"
    / "model_bundle.joblib"
)

CALIBRATOR_PATH = (
    ROOT
    / "artifacts"
    / "calibrator.joblib"
)


Action = Literal[
    "MONITOR",
    "PREPARE_EVIDENCE",
    "MANUAL_REVIEW",
    "RECOMMEND_REFUND",
]

RiskBand = Literal[
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
]


class RiskScoringError(RuntimeError):
    pass


class RiskArtifactError(RiskScoringError):
    pass


class RiskScoreResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    raw_probability: float = Field(
        ge=0,
        le=1,
    )

    calibrated_probability: float = Field(
        ge=0,
        le=1,
    )

    risk_percentage: float = Field(
        ge=0,
        le=100,
    )

    model_recommended_action: Action
    recommended_action: Action

    decision_source: Literal[
        "MERCHANT_ERROR_RULE",
        "COST_OPTIMIZED_MODEL",
    ]

    rules: dict[str, Any]
    risk_band: RiskBand

    expected_costs_rupees: dict[
        Action,
        float,
    ]

    model_version: str
    calibration_version: str
    calibration_method: Literal[
        "isotonic",
        "sigmoid",
    ]

    requires_human_approval: bool

    action_executed: Literal[
        False
    ] = False

    disclosure: Literal[
        "Prediction produced using a calibrated model and deterministic capture-time rules. The model was trained on synthetic data. Recommendations are decision-support signals and no financial action is executed automatically."
    ] = (
        "Prediction produced using a calibrated model "
        "and deterministic capture-time rules. The model "
        "was trained on synthetic data. Recommendations "
        "are decision-support signals and no financial "
        "action is executed automatically."
    )

    @field_validator(
        "raw_probability",
        "calibrated_probability",
        "risk_percentage",
    )
    @classmethod
    def require_finite(
        cls,
        value: float,
    ) -> float:
        if not np.isfinite(value):
            raise ValueError(
                "Risk values must be finite."
            )

        return value


@lru_cache(maxsize=1)
def load_model_bundle() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise RiskArtifactError(
            "artifacts/model_bundle.joblib is missing."
        )

    value = joblib.load(MODEL_PATH)

    if not isinstance(value, dict):
        raise RiskArtifactError(
            "Model bundle is invalid."
        )

    return value


@lru_cache(maxsize=1)
def load_calibrator_bundle() -> dict[str, Any]:
    if not CALIBRATOR_PATH.exists():
        raise RiskArtifactError(
            "artifacts/calibrator.joblib is missing."
        )

    value = joblib.load(CALIBRATOR_PATH)

    if not isinstance(value, dict):
        raise RiskArtifactError(
            "Calibrator bundle is invalid."
        )

    return value


def calibrate_probability(
    raw_probability: np.ndarray,
    calibrator_bundle: dict[str, Any],
) -> np.ndarray:
    method = calibrator_bundle.get("method")
    calibrator = calibrator_bundle.get(
        "calibrator"
    )

    if method == "isotonic":
        result = calibrator.predict(
            raw_probability
        )

    elif method == "sigmoid":
        result = calibrator.predict_proba(
            probability_to_logit(
                raw_probability
            )
        )[:, 1]

    else:
        raise RiskArtifactError(
            "Unknown calibration method."
        )

    result = np.asarray(
        result,
        dtype=float,
    )

    if not np.isfinite(result).all():
        raise RiskArtifactError(
            "Calibrator produced a non-finite value."
        )

    return np.clip(
        result,
        1e-6,
        1 - 1e-6,
    )


def _risk_band(
    probability: float,
) -> RiskBand:
    if probability < 0.01:
        return "LOW"

    if probability < 0.05:
        return "MEDIUM"

    if probability < 0.15:
        return "HIGH"

    return "CRITICAL"


def _prepare_capture_time_features(
    payload: CaptureTimeScoringPayload,
) -> dict[str, Any]:
    request_data = payload.model_dump()

    created_at: datetime = request_data.pop(
        "created_at"
    )

    day_of_week = created_at.weekday()

    request_data.update(
        {
            "hour_of_day": created_at.hour,
            "day_of_week": day_of_week,
            "is_weekend": int(
                day_of_week >= 5
            ),
        }
    )

    forbidden = {
        "payment_id",
        "order_id",
        "customer_id",
        "email",
        "contact",
        "chargeback_within_120d",
        "target",
        "true_fraud",
        "dispute_status",
        "dispute_reason",
        "refund_status",
        "amount_refunded",
    }

    overlap = forbidden & set(request_data)

    if overlap:
        raise RiskScoringError(
            "Forbidden fields reached model input: "
            + ", ".join(sorted(overlap))
        )

    return request_data


def score_capture_time_payment(
    payload: CaptureTimeScoringPayload,
    *,
    model_bundle: dict[str, Any] | None = None,
    calibrator_bundle: dict[str, Any] | None = None,
    cost_params: CostParams | None = None,
) -> RiskScoreResult:
    model_bundle = (
        model_bundle or load_model_bundle()
    )

    calibrator_bundle = (
        calibrator_bundle
        or load_calibrator_bundle()
    )

    request_data = (
        _prepare_capture_time_features(
            payload
        )
    )

    rule_result = evaluate_rules(
        request_data
    )

    model_features = model_bundle.get(
        "model_features"
    )

    if not isinstance(model_features, list):
        raise RiskArtifactError(
            "Model feature list is invalid."
        )

    missing_features = sorted(
        set(model_features)
        - set(request_data)
    )

    if missing_features:
        raise RiskArtifactError(
            "Model input is missing features: "
            + ", ".join(missing_features)
        )

    model_input = pd.DataFrame(
        [request_data]
    )[model_features]

    try:
        transformed_input = model_bundle[
            "preprocessor"
        ].transform(model_input)

        raw_probability = model_bundle[
            "model"
        ].predict_proba(
            transformed_input
        )[:, 1]
    except Exception as error:
        raise RiskArtifactError(
            "Risk model scoring failed."
        ) from error

    raw_probability = np.asarray(
        raw_probability,
        dtype=float,
    )

    if (
        raw_probability.shape != (1,)
        or not np.isfinite(
            raw_probability
        ).all()
    ):
        raise RiskArtifactError(
            "Risk model returned an invalid probability."
        )

    calibrated_probability = (
        calibrate_probability(
            raw_probability,
            calibrator_bundle,
        )
    )

    probability = float(
        calibrated_probability[0]
    )

    amount_rupees = (
        payload.amount_paise / 100
    )

    parameters = (
        cost_params or CostParams()
    )

    expected_costs = (
        calculate_expected_costs(
            np.array([probability]),
            np.array([amount_rupees]),
            parameters,
        )[0]
    )

    selected_index = int(
        np.argmin(expected_costs)
    )

    model_action = ACTIONS[
        selected_index
    ]

    hard_override_action = rule_result[
        "hard_override_action"
    ]

    recommended_action = (
        hard_override_action
        or model_action
    )

    decision_source = (
        "MERCHANT_ERROR_RULE"
        if hard_override_action
        is not None
        else "COST_OPTIMIZED_MODEL"
    )

    cost_breakdown = {
        action: float(
            expected_costs[index]
        )
        for index, action
        in enumerate(ACTIONS)
    }

    requires_human_approval = (
        recommended_action
        in {
            "MANUAL_REVIEW",
            "RECOMMEND_REFUND",
        }
    )

    return RiskScoreResult(
        raw_probability=float(
            raw_probability[0]
        ),
        calibrated_probability=probability,
        risk_percentage=round(
            probability * 100,
            3,
        ),
        model_recommended_action=model_action,
        recommended_action=(
            recommended_action
        ),
        decision_source=decision_source,
        rules=rule_result,
        risk_band=_risk_band(
            probability
        ),
        expected_costs_rupees=(
            cost_breakdown
        ),
        model_version=str(
            model_bundle["model_version"]
        ),
        calibration_version=str(
            calibrator_bundle["version"]
        ),
        calibration_method=(
            calibrator_bundle["method"]
        ),
        requires_human_approval=(
            requires_human_approval
        ),
        action_executed=False,
    )
