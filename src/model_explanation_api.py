from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Literal

import pandas as pd
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from src.explanation_text import DISCLAIMER
from src.model_explanation_delivery import (
    ModelExplanationDeliveryService,
)
from src.model_explanation_generator import (
    EXPECTED_LABEL,
)
from src.model_explanation_service import (
    ExplanationFactor,
    ModelExplanation,
    _validate_factor_sets,
)


ROOT = Path(__file__).resolve().parents[1]

MODEL_EXPLANATIONS_PATH = (
    ROOT
    / "reports"
    / "test_model_explanations.parquet"
)

POLICY_PATH = (
    ROOT
    / "reports"
    / "test_policy.parquet"
)


PAYMENT_ID_PATTERN = re.compile(
    r"^pay_[A-Za-z0-9_-]{3,96}$"
)


Action = Literal[
    "MONITOR",
    "PREPARE_EVIDENCE",
    "MANUAL_REVIEW",
    "RECOMMEND_REFUND",
]


class ModelExplanationRepositoryError(
    RuntimeError
):
    pass


class ModelExplanationNotFoundError(
    ModelExplanationRepositoryError
):
    pass


class TransactionModelExplanationResponse(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    payment_id: str = Field(
        min_length=1,
    )

    calibrated_probability: float = Field(
        ge=0,
        le=1,
    )

    risk_percentage: float = Field(
        ge=0,
        le=100,
    )

    recommended_action: Action

    model_version: str = Field(
        min_length=1,
    )

    shap_explanation_version: str = Field(
        min_length=1,
    )

    deterministic_text_version: str = Field(
        min_length=1,
    )

    delivery_text_version: str = Field(
        min_length=1,
    )

    delivery_mode: Literal[
        "DETERMINISTIC",
        "LIVE_OPENAI",
        "DETERMINISTIC_FALLBACK",
    ]

    provider: Literal[
        "deterministic_formatter",
        "openai",
    ]

    provider_model: str | None = Field(
        default=None,
        max_length=100,
    )

    response_id: str | None = Field(
        default=None,
        max_length=200,
    )

    latency_ms: int = Field(
        ge=0,
    )

    fallback_used: bool

    fallback_reason: str | None = Field(
        default=None,
        pattern=r"^[A-Z][A-Z0-9_]{2,63}$",
    )

    label: Literal[
        "Model explanation — not evidence"
    ]

    explanation: str = Field(
        min_length=1,
    )

    disclaimer: str = Field(
        min_length=1,
    )

    top_positive_factors: list[
        ExplanationFactor
    ] = Field(
        max_length=3,
    )

    top_negative_factors: list[
        ExplanationFactor
    ] = Field(
        max_length=2,
    )

    model_explanation_is_evidence: Literal[
        False
    ] = False

    evidence_copilot_separate: Literal[
        True
    ] = True

    human_approval_required: Literal[
        True
    ] = True

    action_executed: Literal[
        False
    ] = False


def _parse_factor_list(
    raw: object,
) -> list[ExplanationFactor]:
    try:
        payload = json.loads(
            str(raw)
        )
    except json.JSONDecodeError as error:
        raise ModelExplanationRepositoryError(
            "Stored explanation contains "
            "invalid factor JSON"
        ) from error

    if not isinstance(
        payload,
        list,
    ):
        raise ModelExplanationRepositoryError(
            "Stored factor payload must "
            "be a list"
        )

    try:
        return [
            ExplanationFactor.model_validate(
                factor
            )
            for factor in payload
        ]
    except Exception as error:
        raise ModelExplanationRepositoryError(
            "Stored factor payload failed "
            "schema validation"
        ) from error


class ParquetModelExplanationRepository:
    def __init__(
        self,
        *,
        explanation_path: Path = (
            MODEL_EXPLANATIONS_PATH
        ),
        policy_path: Path = POLICY_PATH,
    ) -> None:
        self.explanation_path = Path(
            explanation_path
        )

        self.policy_path = Path(
            policy_path
        )

        self._explanations: (
            pd.DataFrame
            | None
        ) = None

        self._policy: (
            pd.DataFrame
            | None
        ) = None

    def _load(
        self,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
    ]:
        if self._explanations is not None:
            assert self._policy is not None

            return (
                self._explanations,
                self._policy,
            )

        if not self.explanation_path.exists():
            raise ModelExplanationRepositoryError(
                "Stored model explanations "
                "are unavailable"
            )

        if not self.policy_path.exists():
            raise ModelExplanationRepositoryError(
                "Stored policy results "
                "are unavailable"
            )

        explanations = pd.read_parquet(
            self.explanation_path
        )

        policy = pd.read_parquet(
            self.policy_path
        )

        required_explanation_columns = {
            "payment_id",
            "model_version",
            "shap_explanation_version",
            "text_explanation_version",
            "label",
            "explanation",
            "disclaimer",
            "positive_factors_json",
            "negative_factors_json",
        }

        required_policy_columns = {
            "payment_id",
            "calibrated_probability",
            "recommended_action",
        }

        missing_explanation = sorted(
            required_explanation_columns
            - set(
                explanations.columns
            )
        )

        missing_policy = sorted(
            required_policy_columns
            - set(policy.columns)
        )

        if missing_explanation:
            raise ModelExplanationRepositoryError(
                "Stored explanation report is "
                "missing columns: "
                + ", ".join(
                    missing_explanation
                )
            )

        if missing_policy:
            raise ModelExplanationRepositoryError(
                "Stored policy report is "
                "missing columns: "
                + ", ".join(
                    missing_policy
                )
            )

        explanations = (
            explanations.copy()
        )

        policy = policy.copy()

        explanations["payment_id"] = (
            explanations["payment_id"]
            .astype(str)
        )

        policy["payment_id"] = (
            policy["payment_id"]
            .astype(str)
        )

        if explanations[
            "payment_id"
        ].duplicated().any():
            raise ModelExplanationRepositoryError(
                "Stored explanation report "
                "contains duplicate payment IDs"
            )

        if policy[
            "payment_id"
        ].duplicated().any():
            raise ModelExplanationRepositoryError(
                "Stored policy report contains "
                "duplicate payment IDs"
            )

        if set(
            explanations["payment_id"]
        ) != set(
            policy["payment_id"]
        ):
            raise ModelExplanationRepositoryError(
                "Stored explanation and policy "
                "payment IDs are not aligned"
            )

        self._explanations = (
            explanations
        )

        self._policy = policy

        return explanations, policy

    def get(
        self,
        payment_id: str,
    ) -> tuple[
        ModelExplanation,
        float,
        Action,
    ]:
        explanations, policy = (
            self._load()
        )

        explanation_matches = (
            explanations.loc[
                explanations[
                    "payment_id"
                ]
                == payment_id
            ]
        )

        policy_matches = policy.loc[
            policy["payment_id"]
            == payment_id
        ]

        if (
            explanation_matches.empty
            or policy_matches.empty
        ):
            raise (
                ModelExplanationNotFoundError(
                    "Payment not found"
                )
            )

        if (
            len(explanation_matches) != 1
            or len(policy_matches) != 1
        ):
            raise ModelExplanationRepositoryError(
                "Payment matched multiple "
                "stored records"
            )

        row = (
            explanation_matches.iloc[0]
        )

        policy_row = (
            policy_matches.iloc[0]
        )

        positive = _parse_factor_list(
            row[
                "positive_factors_json"
            ]
        )

        negative = _parse_factor_list(
            row[
                "negative_factors_json"
            ]
        )

        _validate_factor_sets(
            positive,
            negative,
        )

        explanation = ModelExplanation(
            payment_id=payment_id,
            model_version=str(
                row["model_version"]
            ),
            shap_explanation_version=str(
                row[
                    "shap_explanation_version"
                ]
            ),
            text_explanation_version=str(
                row[
                    "text_explanation_version"
                ]
            ),
            delivery_mode="deterministic",
            label=str(
                row["label"]
            ),
            explanation=str(
                row["explanation"]
            ),
            disclaimer=str(
                row["disclaimer"]
            ),
            positive_factors=positive,
            negative_factors=negative,
        )

        if (
            explanation.label
            != EXPECTED_LABEL
        ):
            raise ModelExplanationRepositoryError(
                "Stored explanation has an "
                "incorrect safety label"
            )

        if (
            explanation.disclaimer
            != DISCLAIMER
            or DISCLAIMER
            not in explanation.explanation
        ):
            raise ModelExplanationRepositoryError(
                "Stored explanation has an "
                "incorrect disclaimer"
            )

        probability = float(
            policy_row[
                "calibrated_probability"
            ]
        )

        if not 0 <= probability <= 1:
            raise ModelExplanationRepositoryError(
                "Stored calibrated probability "
                "is outside zero to one"
            )

        action = str(
            policy_row[
                "recommended_action"
            ]
        )

        if action not in {
            "MONITOR",
            "PREPARE_EVIDENCE",
            "MANUAL_REVIEW",
            "RECOMMEND_REFUND",
        }:
            raise ModelExplanationRepositoryError(
                "Stored policy action is invalid"
            )

        return (
            explanation,
            probability,
            action,
        )


@lru_cache(maxsize=1)
def get_model_explanation_repository(
) -> ParquetModelExplanationRepository:
    return (
        ParquetModelExplanationRepository()
    )


@lru_cache(maxsize=1)
def get_model_explanation_delivery_service(
) -> ModelExplanationDeliveryService:
    return (
        ModelExplanationDeliveryService()
    )


def _validate_payment_id(
    payment_id: str,
) -> None:
    if not PAYMENT_ID_PATTERN.fullmatch(
        payment_id
    ):
        raise HTTPException(
            status_code=(
                status
                .HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=(
                "Invalid payment ID format."
            ),
        )


router = APIRouter(
    prefix="/api/model-explanations",
    tags=["Model Explanations"],
)


@router.get(
    "/{payment_id}",
    response_model=(
        TransactionModelExplanationResponse
    ),
    responses={
        404: {
            "description": (
                "Held-out payment not found"
            )
        },
        422: {
            "description": (
                "Invalid payment ID"
            )
        },
        503: {
            "description": (
                "Model explanation unavailable"
            )
        },
    },
)
def get_model_explanation(
    payment_id: str,
    use_ai: bool = Query(
        default=False
    ),
    repository: (
        ParquetModelExplanationRepository
    ) = Depends(
        get_model_explanation_repository
    ),
    delivery_service: (
        ModelExplanationDeliveryService
    ) = Depends(
        get_model_explanation_delivery_service
    ),
) -> (
    TransactionModelExplanationResponse
):
    _validate_payment_id(
        payment_id
    )

    try:
        (
            source,
            probability,
            action,
        ) = repository.get(
            payment_id
        )
    except (
        ModelExplanationNotFoundError
    ) as error:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail="Payment not found.",
        ) from error
    except (
        ModelExplanationRepositoryError,
        OSError,
    ) as error:
        raise HTTPException(
            status_code=(
                status
                .HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Model explanation data is "
                "temporarily unavailable."
            ),
        ) from error

    if use_ai:
        try:
            delivered = (
                delivery_service.deliver(
                    source
                )
            )
        except Exception as error:
            raise HTTPException(
                status_code=(
                    status
                    .HTTP_503_SERVICE_UNAVAILABLE
                ),
                detail=(
                    "Model explanation service "
                    "is temporarily unavailable."
                ),
            ) from error

        delivery_mode = (
            delivered.delivery_mode
        )

        provider = delivered.provider
        provider_model = delivered.model
        response_id = (
            delivered.response_id
        )
        latency_ms = delivered.latency_ms
        fallback_used = (
            delivered.fallback_used
        )
        fallback_reason = (
            delivered.fallback_reason
        )
        explanation_text = (
            delivered.explanation
        )
        delivery_text_version = (
            delivered.delivery_text_version
        )
    else:
        delivery_mode = "DETERMINISTIC"
        provider = (
            "deterministic_formatter"
        )
        provider_model = None
        response_id = None
        latency_ms = 0
        fallback_used = False
        fallback_reason = None
        explanation_text = (
            source.explanation
        )
        delivery_text_version = (
            source.text_explanation_version
        )

    return (
        TransactionModelExplanationResponse(
            payment_id=payment_id,
            calibrated_probability=(
                probability
            ),
            risk_percentage=round(
                probability * 100,
                3,
            ),
            recommended_action=action,
            model_version=(
                source.model_version
            ),
            shap_explanation_version=(
                source
                .shap_explanation_version
            ),
            deterministic_text_version=(
                source
                .text_explanation_version
            ),
            delivery_text_version=(
                delivery_text_version
            ),
            delivery_mode=delivery_mode,
            provider=provider,
            provider_model=(
                provider_model
            ),
            response_id=response_id,
            latency_ms=latency_ms,
            fallback_used=fallback_used,
            fallback_reason=(
                fallback_reason
            ),
            label=source.label,
            explanation=(
                explanation_text
            ),
            disclaimer=(
                source.disclaimer
            ),
            top_positive_factors=(
                source.positive_factors[:3]
            ),
            top_negative_factors=(
                source.negative_factors[:2]
            ),
            model_explanation_is_evidence=False,
            evidence_copilot_separate=True,
            human_approval_required=True,
            action_executed=False,
        )
    )