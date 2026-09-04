const API_BASE_URL =
  import.meta.env.VITE_API_URL ??
  "http://127.0.0.1:8000";

export type RecommendedAction =
  | "MONITOR"
  | "PREPARE_EVIDENCE"
  | "MANUAL_REVIEW"
  | "RECOMMEND_REFUND";

export type RiskBand =
  | "LOW"
  | "MEDIUM"
  | "HIGH"
  | "CRITICAL";

export type RazorpayTestOrder = {
  delivery_mode:
    | "CREATED"
    | "IDEMPOTENT_REPLAY";
  checkout_key_id: string;
  order_id: string;
  amount: number;
  currency: "INR";
  receipt: string;
  status: string;
  source: "RAZORPAY_TEST_MODE";
  real_money_used: false;
  action_executed: false;
};

export type CheckoutVerificationResult = {
  verified: true;
  razorpay_order_id: string;
  razorpay_payment_id: string;
  algorithm: "HMAC_SHA256";
  source: "RAZORPAY_TEST_MODE";
  synthetic_evaluation_affected: false;
  financial_action_executed: false;
};

export type MerchantRiskContext = {
  feature_as_of: string;
  card_network: "Visa" | "Mastercard";

  product_category:
    | "fashion"
    | "electronics"
    | "travel"
    | "food"
    | "subscription"
    | "gaming"
    | "education";

  is_digital_good: boolean;
  descriptor_clarity_score: number;
  phone_verified: boolean;
  email_verified: boolean;
  account_age_days: number;
  has_prior_order: 0 | 1;
  total_prior_orders: number;
  prior_disputes_count: number;
  days_since_last_order: number;
  txns_last_1h: number;
  txns_last_24h: number;
  txns_last_7d: number;
  amount_last_24h_paise: number;
  device_is_new: boolean;
  ip_country_matches_billing: boolean;
  ip_is_proxy_or_vpn: boolean;

  cvv_result:
    | "match"
    | "no_match"
    | "not_provided";

  threeds_status:
    | "authenticated"
    | "attempted"
    | "not_authenticated";

  threeds_liability_shift: boolean;
  billing_shipping_distance_km: number;
  is_duplicate_payment: boolean;
  cancelled_subscription_billed: boolean;
};

export type RiskScore = {
  raw_probability: number;
  calibrated_probability: number;
  risk_percentage: number;
  model_recommended_action: RecommendedAction;
  recommended_action: RecommendedAction;
  decision_source: string;
  rules: unknown;
  risk_band: RiskBand;
  explanation: unknown;
  expected_costs_rupees: Record<
    string,
    number
  >;
  model_version: string;
  calibration_method: string;
  requires_human_approval: boolean;
  action_executed: false;
  disclosure: string;
};

export type RazorpayPaymentRiskResult = {
  delivery_mode:
    | "CALCULATED"
    | "IDEMPOTENT_REPLAY";

  razorpay_payment_id: string;
  razorpay_order_id: string;

  provider_status:
    | "authorized"
    | "captured";

  provider_method: "card";

  normalization_version:
    "razorpay-normalizer-v1";

  score: RiskScore;
  source: "RAZORPAY_TEST_MODE";
  synthetic_evaluation_affected: false;
  human_approval_required: true;
  financial_action_executed: false;
  disclosure: string;
};

export type VerifiedPaymentRiskResult = {
  verification: CheckoutVerificationResult;
  risk_result: RazorpayPaymentRiskResult;
  source: "RAZORPAY_TEST_MODE";
  signature_required_before_scoring: true;
  synthetic_evaluation_affected: false;
  financial_action_executed: false;
};

type SafeErrorPayload = {
  detail?: string;
};

async function requestJson<T>(
  path: string,
  options: RequestInit,
): Promise<T> {
  const response = await fetch(
    `${API_BASE_URL}${path}`,
    options,
  );

  if (!response.ok) {
    let message =
      `Request failed with status ${response.status}`;

    try {
      const payload =
        (await response.json()) as SafeErrorPayload;

      if (payload.detail) {
        message = payload.detail;
      }
    } catch {
      // Keep the safe HTTP status message.
    }

    throw new Error(message);
  }

  return response.json() as Promise<T>;
}

function newRequestId(): string {
  const identifier =
    globalThis.crypto
      .randomUUID()
      .replace(/-/g, "");

  return `checkout_${identifier}`;
}

export function createRazorpayTestOrder(
  amountPaise: number,
): Promise<RazorpayTestOrder> {
  return requestJson<RazorpayTestOrder>(
    "/api/razorpay-test/orders",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Idempotency-Key": newRequestId(),
      },
      body: JSON.stringify({
        amount: amountPaise,
        currency: "INR",
        receipt: `radar_${Date.now()}`,
      }),
    },
  );
}

export function verifyRazorpayCheckout(
  orderId: string,
  paymentId: string,
  signature: string,
): Promise<CheckoutVerificationResult> {
  return requestJson<CheckoutVerificationResult>(
    (
      `/api/razorpay-test/orders/` +
      `${encodeURIComponent(orderId)}/` +
      "verify-checkout"
    ),
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        razorpay_payment_id: paymentId,
        razorpay_signature: signature,
      }),
    },
  );
}

export function verifyAndScoreRazorpayCheckout(
  orderId: string,
  paymentId: string,
  signature: string,
  context: MerchantRiskContext,
): Promise<VerifiedPaymentRiskResult> {
  return requestJson<VerifiedPaymentRiskResult>(
    (
      `/api/razorpay-test/orders/` +
      `${encodeURIComponent(orderId)}/` +
      "verify-and-score"
    ),
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        checkout: {
          razorpay_payment_id: paymentId,
          razorpay_signature: signature,
        },
        context,
      }),
    },
  );
}