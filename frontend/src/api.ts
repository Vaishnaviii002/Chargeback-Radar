const API_BASE_URL =
  import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

export type PolicyParams = {
  chargeback_fee: number;
  gross_margin_rate: number;
  evidence_cost: number;
  manual_review_cost: number;
  customer_friction_rate: number;
  evidence_recovery_rate: number;
  review_prevention_rate: number;
  refund_cost_rate: number;
  risk_program_penalty: number;
};

export type PolicyResult = {
  parameters: PolicyParams;
  records_evaluated: number;
  action_mix: Record<string, number>;
  intervention_count: number;
  intervention_rate: number;
  precision: number;
  recall: number;
  confusion_matrix: {
    true_negative: number;
    false_positive: number;
    false_negative: number;
    true_positive: number;
  };
  costs: {
    do_nothing_baseline_rupees: number;
    policy_cost_rupees: number;
    false_positive_cost_rupees: number;
    estimated_net_benefit_rupees: number;
  };
  disclosure: string;
};

export type MetricsResponse = {
  evaluation: {
    dataset: {
      test_rows: number;
      positive_labels: number;
      base_rate: number;
    };
    operating_policy: {
      review_capacity: number;
      operating_threshold: number;
      flagged_count: number;
      flagged_rate: number;
    };
    model_performance: {
      average_precision: number;
      precision: number;
      recall: number;
      f1: number;
      precision_at_1_percent: number;
      precision_at_5_percent: number;
      expected_calibration_error: number;
    };
    confusion_matrix: {
      true_negative: number;
      false_positive: number;
      false_negative: number;
      true_positive: number;
    };
    false_positive_cost: {
      manual_review_cost_per_case: number;
      false_positive_count: number;
      total_rupees: number;
    };
    exposure: {
      assumed_chargeback_fee_rupees: number;
      detected_chargeback_exposure_rupees: number;
      missed_chargeback_exposure_rupees: number;
    };
    disclosures: {
      synthetic_data: boolean;
      split: string;
      money_note: string;
    };
  };
  calibration: {
    selected_method: string;
    test_raw_brier: number;
    test_calibrated_brier: number;
    test_raw_average_precision: number;
    test_calibrated_average_precision: number;
  };
  model: {
    model_version: string;
    training_rows: number;
    calibration_rows: number;
    test_rows: number;
    best_iteration: number;
  };
  default_policy: PolicyResult;
};

export type PrecisionRecallPoint = {
  precision: number;
  recall: number;
  threshold: number | null;
};

export type CalibrationPoint = {
  predicted_probability: number;
  observed_rate: number;
};

export type CurvesResponse = {
  precision_recall: {
    base_rate: number;
    operating_threshold: number;
    points: PrecisionRecallPoint[];
  };
  calibration: CalibrationPoint[];
};

export type Transaction = {
  payment_id: string;
  customer_id: string;
  created_at: string;
  amount_paise: number;
  calibrated_probability: number;
  recommended_action: string;
  is_duplicate_payment: boolean;
  cancelled_subscription_billed: boolean;
  chargeback_within_120d: number;
};

export type TransactionsResponse = {
  count: number;
  transactions: Transaction[];
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
    const message = await response.text();

    throw new Error(
      message || `Request failed with status ${response.status}`,
    );
  }

  return response.json() as Promise<T>;
}

export function fetchMetrics(): Promise<MetricsResponse> {
  return requestJson<MetricsResponse>("/api/metrics");
}

export function fetchCurves(): Promise<CurvesResponse> {
  return requestJson<CurvesResponse>("/api/curves");
}

export function fetchTransactions(
  limit = 50,
): Promise<TransactionsResponse> {
  return requestJson<TransactionsResponse>(
    `/api/transactions?limit=${limit}`,
  );
}

export function fetchTransaction(
  paymentId: string,
): Promise<Record<string, unknown>> {
  return requestJson<Record<string, unknown>>(
    `/api/transactions/${paymentId}`,
  );
}

export function simulatePolicy(
  params: PolicyParams,
): Promise<PolicyResult> {
  return requestJson<PolicyResult>("/api/simulate", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

export function scorePayment(
  payment: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  return requestJson<Record<string, unknown>>("/api/score", {
    method: "POST",
    body: JSON.stringify(payment),
  });
}