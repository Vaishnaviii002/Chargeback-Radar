import {
  useState,
} from "react";

import {
  createRazorpayTestOrder,
  verifyAndScoreRazorpayCheckout,
  type MerchantRiskContext,
  type RazorpayTestOrder,
  type VerifiedPaymentRiskResult,
} from "../razorpayTestApi";

type CheckoutSuccessResponse = {
  razorpay_payment_id: string;
  razorpay_order_id: string;
  razorpay_signature: string;
};

type CheckoutOptions = {
  key: string;
  amount: number;
  currency: string;
  name: string;
  description: string;
  order_id: string;
  handler: (
    response: CheckoutSuccessResponse,
  ) => void;
  modal: {
    ondismiss: () => void;
  };
  theme: {
    color: string;
  };
};

type CheckoutInstance = {
  open: () => void;
  on: (
    event: "payment.failed",
    handler: () => void,
  ) => void;
};

type CheckoutConstructor = new (
  options: CheckoutOptions,
) => CheckoutInstance;

declare global {
  interface Window {
    Razorpay?: CheckoutConstructor;
  }
}

type CheckoutPhase =
  | "IDLE"
  | "CREATING_ORDER"
  | "CHECKOUT_OPEN"
  | "VERIFYING_AND_SCORING"
  | "COMPLETE"
  | "ERROR";

type DemoProfile =
  | "NORMAL"
  | "HEIGHTENED";

let checkoutScriptPromise:
  Promise<void> | null = null;

function loadCheckoutScript(): Promise<void> {
  if (window.Razorpay) {
    return Promise.resolve();
  }

  if (checkoutScriptPromise) {
    return checkoutScriptPromise;
  }

  checkoutScriptPromise = new Promise(
    (resolve, reject) => {
      const existing = document.querySelector(
        'script[data-chargeback-radar-razorpay="true"]',
      ) as HTMLScriptElement | null;

      if (existing) {
        existing.addEventListener(
          "load",
          () => resolve(),
          { once: true },
        );

        existing.addEventListener(
          "error",
          () => reject(
            new Error(
              "Razorpay Checkout could not be loaded.",
            ),
          ),
          { once: true },
        );

        return;
      }

      const script =
        document.createElement("script");

      script.src =
        "https://checkout.razorpay.com/v1/checkout.js";

      script.async = true;

      script.dataset.chargebackRadarRazorpay =
        "true";

      script.onload = () => resolve();

      script.onerror = () => reject(
        new Error(
          "Razorpay Checkout could not be loaded.",
        ),
      );

      document.body.appendChild(script);
    },
  );

  return checkoutScriptPromise;
}

function captureTimestamp(): string {
  const wholeSecond =
    Math.floor(Date.now() / 1000) * 1000;

  return new Date(
    wholeSecond,
  ).toISOString();
}

function buildContext(
  profile: DemoProfile,
  featureAsOf: string,
  amountPaise: number,
): MerchantRiskContext {
  if (profile === "HEIGHTENED") {
    return {
      feature_as_of: featureAsOf,
      card_network: "Visa",
      product_category: "subscription",
      is_digital_good: true,
      descriptor_clarity_score: 0.3,
      phone_verified: false,
      email_verified: false,
      account_age_days: 2,
      has_prior_order: 0,
      total_prior_orders: 0,
      prior_disputes_count: 0,
      days_since_last_order: 999,
      txns_last_1h: 4,
      txns_last_24h: 8,
      txns_last_7d: 10,
      amount_last_24h_paise:
        amountPaise * 6,
      device_is_new: true,
      ip_country_matches_billing: false,
      ip_is_proxy_or_vpn: true,
      cvv_result: "no_match",
      threeds_status: "not_authenticated",
      threeds_liability_shift: false,
      billing_shipping_distance_km: 2200,
      is_duplicate_payment: false,
      cancelled_subscription_billed: false,
    };
  }

  return {
    feature_as_of: featureAsOf,
    card_network: "Visa",
    product_category: "electronics",
    is_digital_good: false,
    descriptor_clarity_score: 0.8,
    phone_verified: true,
    email_verified: true,
    account_age_days: 365,
    has_prior_order: 1,
    total_prior_orders: 5,
    prior_disputes_count: 0,
    days_since_last_order: 30,
    txns_last_1h: 0,
    txns_last_24h: 0,
    txns_last_7d: 1,
    amount_last_24h_paise: 0,
    device_is_new: false,
    ip_country_matches_billing: true,
    ip_is_proxy_or_vpn: false,
    cvv_result: "match",
    threeds_status: "authenticated",
    threeds_liability_shift: true,
    billing_shipping_distance_km: 10,
    is_duplicate_payment: false,
    cancelled_subscription_billed: false,
  };
}

function formatMoney(
  amountPaise: number,
): string {
  return new Intl.NumberFormat(
    "en-IN",
    {
      style: "currency",
      currency: "INR",
    },
  ).format(amountPaise / 100);
}

function formatAction(
  action: string,
): string {
  return action
    .toLowerCase()
    .split("_")
    .map(
      (word) =>
        word.charAt(0).toUpperCase() +
        word.slice(1),
    )
    .join(" ");
}

export default function RazorpayTestCheckout() {
  const [amountRupees, setAmountRupees] =
    useState(100);

  const [profile, setProfile] =
    useState<DemoProfile>("NORMAL");

  const [phase, setPhase] =
    useState<CheckoutPhase>("IDLE");

  const [order, setOrder] =
    useState<RazorpayTestOrder | null>(null);

  const [result, setResult] =
    useState<VerifiedPaymentRiskResult | null>(
      null,
    );

  const [error, setError] = useState("");

  const busy = [
    "CREATING_ORDER",
    "CHECKOUT_OPEN",
    "VERIFYING_AND_SCORING",
  ].includes(phase);

  async function startTestCheckout() {
    setError("");
    setResult(null);
    setOrder(null);

    const amountPaise =
      Math.round(amountRupees * 100);

    if (
      !Number.isInteger(amountPaise) ||
      amountPaise < 100 ||
      amountPaise > 10_000_000
    ) {
      setPhase("ERROR");
      setError(
        "Enter an amount between ₹1 and ₹1,00,000.",
      );
      return;
    }

    const featureAsOf =
      captureTimestamp();

    const context = buildContext(
      profile,
      featureAsOf,
      amountPaise,
    );

    try {
      setPhase("CREATING_ORDER");

      await loadCheckoutScript();

      const createdOrder =
        await createRazorpayTestOrder(
          amountPaise,
        );

      setOrder(createdOrder);

      const Checkout = window.Razorpay;

      if (!Checkout) {
        throw new Error(
          "Razorpay Checkout is unavailable.",
        );
      }

      const checkout = new Checkout({
        key: createdOrder.checkout_key_id,
        amount: createdOrder.amount,
        currency: createdOrder.currency,
        name: "Chargeback Radar",
        description:
          "AI Risk Manager — Razorpay Test Mode",
        order_id: createdOrder.order_id,

        handler: (
          checkoutResponse,
        ) => {
          void verifyAndScore(
            createdOrder,
            checkoutResponse,
            context,
          );
        },

        modal: {
          ondismiss: () => {
            setPhase((currentPhase) =>
              currentPhase === "CHECKOUT_OPEN"
                ? "IDLE"
                : currentPhase,
            );
          },
        },

        theme: {
          color: "#6d5dfc",
        },
      });

      checkout.on(
        "payment.failed",
        () => {
          setPhase("ERROR");
          setError(
            "The Test Mode payment was not completed.",
          );
        },
      );

      setPhase("CHECKOUT_OPEN");
      checkout.open();

    } catch (caughtError) {
      setPhase("ERROR");

      setError(
        caughtError instanceof Error
          ? caughtError.message
          : "Test Mode Checkout failed safely.",
      );
    }
  }

  async function verifyAndScore(
    createdOrder: RazorpayTestOrder,
    checkoutResponse: CheckoutSuccessResponse,
    context: MerchantRiskContext,
  ) {
    try {
      setPhase("VERIFYING_AND_SCORING");

      if (
        checkoutResponse.razorpay_order_id !==
        createdOrder.order_id
      ) {
        throw new Error(
          "Checkout returned an unexpected order.",
        );
      }

      const completed =
        await verifyAndScoreRazorpayCheckout(
          createdOrder.order_id,
          checkoutResponse.razorpay_payment_id,
          checkoutResponse.razorpay_signature,
          context,
        );

      setResult(completed);
      setPhase("COMPLETE");

    } catch (caughtError) {
      setPhase("ERROR");

      setError(
        caughtError instanceof Error
          ? caughtError.message
          : (
              "Payment verification and scoring " +
              "failed safely."
            ),
      );
    }
  }

  const score = result?.risk_result.score;

  return (
    <section className="razorpay-test-panel">
      <div className="razorpay-test-heading">
        <div>
          <span className="test-mode-label">
            Razorpay Test Mode
          </span>

          <h2>Live payment-risk demonstration</h2>

          <p>
            Create a Test Mode card payment,
            authenticate its Checkout signature and
            score it with the calibrated risk model.
          </p>
        </div>

        <span className="no-real-money">
          No real money
        </span>
      </div>

      <div className="razorpay-test-controls">
        <label>
          Test amount in rupees

          <input
            type="number"
            min="1"
            max="100000"
            step="1"
            value={amountRupees}
            disabled={busy}
            onChange={(event) => {
              setAmountRupees(
                Number(event.target.value),
              );
            }}
          />
        </label>

        <label>
          Merchant capture-time profile

          <select
            value={profile}
            disabled={busy}
            onChange={(event) => {
              setProfile(
                event.target.value as DemoProfile,
              );
            }}
          >
            <option value="NORMAL">
              Normal signals
            </option>

            <option value="HEIGHTENED">
              Heightened-risk signals
            </option>
          </select>
        </label>

        <button
          type="button"
          disabled={busy}
          onClick={() => {
            void startTestCheckout();
          }}
        >
          {phase === "CREATING_ORDER"
            ? "Creating test order…"
            : phase === "CHECKOUT_OPEN"
              ? "Checkout open…"
              : phase ===
                  "VERIFYING_AND_SCORING"
                ? "Verifying and scoring…"
                : "Create test payment"}
        </button>
      </div>

      <p className="capture-profile-note">
        The selected profile represents explicitly
        declared merchant-side capture-time facts.
        It is not information supplied by Razorpay.
      </p>

      {order && (
        <div className="razorpay-test-order">
          <span>Order</span>
          <strong>{order.order_id}</strong>
          <span>
            {formatMoney(order.amount)}
          </span>
        </div>
      )}

      {phase === "COMPLETE" && result && score && (
        <div
          className="razorpay-score-result"
          role="status"
        >
          <div className="score-verification-line">
            <strong>
              Checkout signature verified
            </strong>

            <span>
              {result.verification.algorithm}
            </span>
          </div>

          <div className="razorpay-score-grid">
            <div>
              <span>Calibrated risk</span>
              <strong>
                {score.risk_percentage.toFixed(3)}%
              </strong>
            </div>

            <div>
              <span>Risk band</span>
              <strong>{score.risk_band}</strong>
            </div>

            <div>
              <span>Recommended action</span>
              <strong>
                {formatAction(
                  score.recommended_action,
                )}
              </strong>
            </div>

            <div>
              <span>Payment status</span>
              <strong>
                {result.risk_result.provider_status}
              </strong>
            </div>
          </div>

          <div className="razorpay-score-metadata">
            <span>
              Payment:{" "}
              {
                result.verification
                  .razorpay_payment_id
              }
            </span>

            <span>
              Model {score.model_version} ·{" "}
              {score.calibration_method} calibration
            </span>

            <span>
              Human approval: required
            </span>

            <span>
              Action executed: no
            </span>
          </div>
        </div>
      )}

      {phase === "ERROR" && error && (
        <div
          className="razorpay-test-error"
          role="alert"
        >
          {error}
        </div>
      )}

      <p className="razorpay-test-disclosure">
        This Razorpay Test Mode result is separate
        from the synthetic held-out evaluation.
        Model output is decision support—not proof
        of fraud, customer intent or a future
        chargeback. No refund, dispute response or
        customer message is executed automatically.
      </p>
    </section>
  );
}