const API_BASE_URL =
  import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

export type RecommendedAction =
  | "MONITOR"
  | "PREPARE_EVIDENCE"
  | "MANUAL_REVIEW"
  | "RECOMMEND_REFUND";

export type ExplanationDirection =
  | "increases_risk"
  | "decreases_risk";

export type ExplanationFactor = {
  feature: string;
  value: unknown;
  shap_value: number;
  direction: ExplanationDirection;
};

export type ModelExplanationResponse = {
  payment_id: string;
  calibrated_probability: number;
  risk_percentage: number;
  recommended_action: RecommendedAction;

  model_version: string;
  shap_explanation_version: string;
  deterministic_text_version: string;
  delivery_text_version: string;

  delivery_mode:
    | "DETERMINISTIC"
    | "LIVE_OPENAI"
    | "CACHE"
    | "DETERMINISTIC_FALLBACK";

  provider: string;
  provider_model: string | null;
  response_id: string | null;
  latency_ms: number;
  fallback_used: boolean;
  fallback_reason: string | null;

  label: "Model explanation — not evidence";
  explanation: string;
  disclaimer: string;

  top_positive_factors: ExplanationFactor[];
  top_negative_factors: ExplanationFactor[];

  model_explanation_is_evidence: false;
  evidence_copilot_separate: true;
  human_approval_required: true;
  action_executed: false;
};

export type ExplanationTransaction = {
  payment_id: string;
  calibrated_probability: number;
  recommended_action: RecommendedAction;
};

type TransactionApiResponse =
  | ExplanationTransaction[]
  | {
      count: number;
      transactions: ExplanationTransaction[];
    };

async function requestJson<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;

    try {
      const payload = (await response.json()) as {
        detail?: string;
      };

      if (payload.detail) {
        message = payload.detail;
      }
    } catch {
      // Preserve the HTTP status fallback.
    }

    throw new Error(message);
  }

  return response.json() as Promise<T>;
}

export async function fetchExplanationTransactions(
  limit = 50,
): Promise<ExplanationTransaction[]> {
  const result = await requestJson<TransactionApiResponse>(
    `/api/transactions?limit=${limit}`,
  );

  return Array.isArray(result) ? result : result.transactions;
}

export function fetchModelExplanation(
  paymentId: string,
  useAI = false,
): Promise<ModelExplanationResponse> {
  return requestJson<ModelExplanationResponse>(
    `/api/model-explanations/${encodeURIComponent(
      paymentId,
    )}?use_ai=${useAI}`,
  );
}