import { useCallback, useEffect, useState } from "react";
import {
  BrainCircuit,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  TrendingDown,
  TrendingUp,
} from "lucide-react";

import {
  fetchExplanationTransactions,
  fetchModelExplanation,
  type ExplanationFactor,
  type ExplanationTransaction,
  type ModelExplanationResponse,
} from "../modelExplanationApi";

function formatFeatureName(feature: string) {
  return feature
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatValue(value: unknown) {
  if (typeof value === "boolean") {
    return value ? "Yes" : "No";
  }

  if (value === null || value === undefined) {
    return "Not available";
  }

  if (typeof value === "object") {
    return JSON.stringify(value);
  }

  return String(value);
}

function formatAction(action: string) {
  return action
    .toLowerCase()
    .split("_")
    .map(
      (word) =>
        word.charAt(0).toUpperCase() + word.slice(1),
    )
    .join(" ");
}

function FactorList({
  factors,
  kind,
}: {
  factors: ExplanationFactor[];
  kind: "positive" | "negative";
}) {
  const Icon =
    kind === "positive" ? TrendingUp : TrendingDown;

  return (
    <div
      className={`model-factor-group model-factor-${kind}`}
    >
      <div className="model-factor-heading">
        <Icon size={18} />

        <div>
          <strong>
            {kind === "positive"
              ? "Risk-increasing contributions"
              : "Risk-reducing contributions"}
          </strong>

          <span>
            Contributions to the model estimate—not causes
          </span>
        </div>
      </div>

      {factors.length === 0 ? (
        <p className="model-factor-empty">
          No factors available.
        </p>
      ) : (
        <div className="model-factor-list">
          {factors.map((factor) => (
            <article
              className="model-factor-card"
              key={`${kind}-${factor.feature}`}
            >
              <div className="model-factor-title">
                <strong>
                  {formatFeatureName(factor.feature)}
                </strong>

                <span
                  className={`model-direction model-direction-${kind}`}
                >
                  {kind === "positive"
                    ? "Increases estimate"
                    : "Decreases estimate"}
                </span>
              </div>

              <div className="model-factor-details">
                <span>
                  Observed value
                  <strong>{formatValue(factor.value)}</strong>
                </span>

                <span>
                  SHAP contribution
                  <strong>
                    {factor.shap_value > 0 ? "+" : ""}
                    {factor.shap_value.toFixed(4)}
                  </strong>
                </span>
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}

export default function ModelExplanationWorkbench() {
  const [transactions, setTransactions] = useState<
    ExplanationTransaction[]
  >([]);

  const [selectedPaymentId, setSelectedPaymentId] =
    useState("");

  const [explanation, setExplanation] =
    useState<ModelExplanationResponse | null>(null);

  const [loading, setLoading] = useState(true);
  const [wordingLoading, setWordingLoading] =
    useState(false);

  const [error, setError] = useState("");

  const loadExplanation = useCallback(
    async (
      paymentId: string,
      useAI: boolean,
    ) => {
      try {
        if (useAI) {
          setWordingLoading(true);
        } else {
          setLoading(true);
        }

        setError("");

        const result = await fetchModelExplanation(
          paymentId,
          useAI,
        );

        setExplanation(result);
      } catch (requestError) {
        setError(
          requestError instanceof Error
            ? requestError.message
            : "Could not load the model explanation.",
        );
      } finally {
        setLoading(false);
        setWordingLoading(false);
      }
    },
    [],
  );

  useEffect(() => {
    let active = true;

    async function initialise() {
      try {
        setLoading(true);
        setError("");

        const result =
          await fetchExplanationTransactions(50);

        if (!active) {
          return;
        }

        setTransactions(result);

        if (result.length === 0) {
          throw new Error(
            "No held-out transactions are available.",
          );
        }

        const initialPaymentId = result[0].payment_id;

        setSelectedPaymentId(initialPaymentId);

        const initialExplanation =
          await fetchModelExplanation(
            initialPaymentId,
            false,
          );

        if (active) {
          setExplanation(initialExplanation);
        }
      } catch (requestError) {
        if (active) {
          setError(
            requestError instanceof Error
              ? requestError.message
              : "Could not initialise model explanations.",
          );
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    void initialise();

    return () => {
      active = false;
    };
  }, []);

  function selectPayment(paymentId: string) {
    setSelectedPaymentId(paymentId);
    void loadExplanation(paymentId, false);
  }

  return (
    <section
      className="model-explanation-workbench"
      aria-labelledby="model-explanation-title"
    >
      <div className="model-explanation-header">
        <div className="model-explanation-title">
          <div className="model-explanation-icon">
            <BrainCircuit size={24} />
          </div>

          <div>
            <span className="section-eyebrow">
              Explainable risk model
            </span>

            <h2 id="model-explanation-title">
              Transaction model explanation
            </h2>

            <p>
              Validated SHAP contributions from the held-out
              model evaluation.
            </p>
          </div>
        </div>

        <div className="model-explanation-controls">
          <label htmlFor="model-explanation-payment">
            Held-out payment
          </label>

          <select
            id="model-explanation-payment"
            value={selectedPaymentId}
            disabled={loading || transactions.length === 0}
            onChange={(event) =>
              selectPayment(event.target.value)
            }
          >
            {transactions.map((transaction) => (
              <option
                value={transaction.payment_id}
                key={transaction.payment_id}
              >
                {transaction.payment_id}
              </option>
            ))}
          </select>
        </div>
      </div>

      {error ? (
        <div
          className="model-explanation-error"
          role="alert"
        >
          <strong>Explanation unavailable</strong>
          <span>{error}</span>

          <button
            type="button"
            disabled={!selectedPaymentId}
            onClick={() =>
              void loadExplanation(
                selectedPaymentId,
                false,
              )
            }
          >
            <RefreshCw size={15} />
            Retry
          </button>
        </div>
      ) : null}

      {loading ? (
        <div
          className="model-explanation-loading"
          aria-live="polite"
        >
          <RefreshCw size={20} className="spinning-icon" />
          Loading validated model factors…
        </div>
      ) : null}

      {!loading && explanation ? (
        <>
          <div className="model-explanation-summary">
            <div>
              <span>Calibrated probability</span>
              <strong>
                {explanation.risk_percentage.toFixed(3)}%
              </strong>
            </div>

            <div>
              <span>Recommended action</span>
              <strong>
                {formatAction(
                  explanation.recommended_action,
                )}
              </strong>
            </div>

            <div>
              <span>Model version</span>
              <strong>{explanation.model_version}</strong>
            </div>

            <div>
              <span>Explanation version</span>
              <strong>
                {explanation.shap_explanation_version}
              </strong>
            </div>
          </div>

          <div className="model-explanation-notice">
            <ShieldCheck size={20} />

            <div>
              <strong>{explanation.label}</strong>
              <p>{explanation.disclaimer}</p>
            </div>
          </div>

          <div className="model-explanation-copy">
            <div className="model-explanation-copy-heading">
              <div>
                <span>Analyst-readable explanation</span>
                <small>
                  {explanation.delivery_mode} ·{" "}
                  {explanation.provider} ·{" "}
                  {explanation.latency_ms} ms
                </small>
              </div>

              <button
                type="button"
                className="model-ai-wording-button"
                disabled={
                  wordingLoading || !selectedPaymentId
                }
                onClick={() =>
                  void loadExplanation(
                    selectedPaymentId,
                    true,
                  )
                }
              >
                {wordingLoading ? (
                  <RefreshCw
                    size={15}
                    className="spinning-icon"
                  />
                ) : (
                  <Sparkles size={15} />
                )}

                {wordingLoading
                  ? "Generating…"
                  : "Improve wording safely"}
              </button>
            </div>

            <p>{explanation.explanation}</p>

            {explanation.fallback_used ? (
              <small className="model-fallback-note">
                Deterministic fallback used:{" "}
                {explanation.fallback_reason ??
                  "AI wording was unavailable."}
              </small>
            ) : null}
          </div>

          <div className="model-factor-grid">
            <FactorList
              kind="positive"
              factors={explanation.top_positive_factors}
            />

            <FactorList
              kind="negative"
              factors={explanation.top_negative_factors}
            />
          </div>

          <div className="model-explanation-governance">
            <ShieldCheck size={18} />

            <span>
              This panel explains model contributions only.
              Evidence Copilot uses separately validated
              operational facts. Human approval is required and
              no action has been executed.
            </span>
          </div>
        </>
      ) : null}
    </section>
  );
}