import { useState } from "react";

const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

type Parameters = {
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

type SimulationResult = {
  records_evaluated: number;
  intervention_count: number;
  intervention_rate: number;
  precision: number;
  recall: number;
  false_positive_cost?: number;
  estimated_net_benefit?: number;
  net_benefit?: number;
  action_mix: Record<string, number>;
  confusion_matrix: {
    true_positive: number;
    false_positive: number;
    false_negative: number;
    true_negative: number;
  };
};

const defaults: Parameters = {
  chargeback_fee: 1500,
  gross_margin_rate: 0.35,
  evidence_cost: 40,
  manual_review_cost: 150,
  customer_friction_rate: 0.08,
  evidence_recovery_rate: 0.45,
  review_prevention_rate: 0.65,
  refund_cost_rate: 0.35,
  risk_program_penalty: 800,
};

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatMoney(value: number) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(value);
}

export default function PolicySimulator() {
  const [parameters, setParameters] = useState(defaults);
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function updateParameter(
    name: keyof Parameters,
    value: number
  ) {
    setParameters((current) => ({
      ...current,
      [name]: value,
    }));
  }

  async function runSimulation() {
    try {
      setLoading(true);
      setError("");

      const response = await fetch(`${API_URL}/api/simulate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(parameters),
      });

      if (!response.ok) {
        throw new Error(`Simulation failed with ${response.status}`);
      }

      const data: SimulationResult = await response.json();
      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not run simulation"
      );
    } finally {
      setLoading(false);
    }
  }

  const netBenefit =
    result?.estimated_net_benefit ?? result?.net_benefit;

  return (
    <section className="simulator-panel">
      <div className="simulator-heading">
        <div>
          <span className="section-kicker">Interactive policy lab</span>
          <h2>False-positive cost simulator</h2>
          <p>
            Change business assumptions and recalculate the lowest-cost
            defensive actions.
          </p>
        </div>

        <button
          className="simulate-button"
          onClick={runSimulation}
          disabled={loading}
        >
          {loading ? "Calculating…" : "Run simulation"}
        </button>
      </div>

      <div className="simulator-content">
        <div className="slider-grid">
          <label className="slider-control">
            <div>
              <span>Chargeback fee</span>
              <strong>{formatMoney(parameters.chargeback_fee)}</strong>
            </div>

            <input
              type="range"
              min="0"
              max="3000"
              step="100"
              value={parameters.chargeback_fee}
              onChange={(event) =>
                updateParameter(
                  "chargeback_fee",
                  Number(event.target.value)
                )
              }
            />
          </label>

          <label className="slider-control">
            <div>
              <span>Manual-review cost</span>
              <strong>{formatMoney(parameters.manual_review_cost)}</strong>
            </div>

            <input
              type="range"
              min="0"
              max="600"
              step="25"
              value={parameters.manual_review_cost}
              onChange={(event) =>
                updateParameter(
                  "manual_review_cost",
                  Number(event.target.value)
                )
              }
            />
          </label>

          <label className="slider-control">
            <div>
              <span>Evidence preparation cost</span>
              <strong>{formatMoney(parameters.evidence_cost)}</strong>
            </div>

            <input
              type="range"
              min="0"
              max="300"
              step="10"
              value={parameters.evidence_cost}
              onChange={(event) =>
                updateParameter(
                  "evidence_cost",
                  Number(event.target.value)
                )
              }
            />
          </label>

          <label className="slider-control">
            <div>
              <span>Customer friction</span>
              <strong>
                {formatPercent(parameters.customer_friction_rate)}
              </strong>
            </div>

            <input
              type="range"
              min="0"
              max="0.3"
              step="0.01"
              value={parameters.customer_friction_rate}
              onChange={(event) =>
                updateParameter(
                  "customer_friction_rate",
                  Number(event.target.value)
                )
              }
            />
          </label>

          <label className="slider-control">
            <div>
              <span>Evidence recovery rate</span>
              <strong>
                {formatPercent(parameters.evidence_recovery_rate)}
              </strong>
            </div>

            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={parameters.evidence_recovery_rate}
              onChange={(event) =>
                updateParameter(
                  "evidence_recovery_rate",
                  Number(event.target.value)
                )
              }
            />
          </label>

          <label className="slider-control">
            <div>
              <span>Review prevention rate</span>
              <strong>
                {formatPercent(parameters.review_prevention_rate)}
              </strong>
            </div>

            <input
              type="range"
              min="0.1"
              max="0.95"
              step="0.05"
              value={parameters.review_prevention_rate}
              onChange={(event) =>
                updateParameter(
                  "review_prevention_rate",
                  Number(event.target.value)
                )
              }
            />
          </label>
        </div>

        <div className="simulation-results">
          {!result && (
            <div className="simulation-empty">
              <strong>Ready to simulate</strong>
              <p>
                Adjust assumptions and run the policy to see its effect.
              </p>
            </div>
          )}

          {error && <p className="simulation-error">{error}</p>}

          {result && (
            <>
              <div className="result-card">
                <span>Intervention rate</span>
                <strong>
                  {formatPercent(result.intervention_rate)}
                </strong>
                <small>
                  {result.intervention_count.toLocaleString()} payments
                </small>
              </div>

              <div className="result-card">
                <span>Intervention precision</span>
                <strong>{formatPercent(result.precision)}</strong>
                <small>Chargebacks among interventions</small>
              </div>

              <div className="result-card">
                <span>Chargeback recall</span>
                <strong>{formatPercent(result.recall)}</strong>
                <small>Known chargebacks covered</small>
              </div>

              {netBenefit !== undefined && (
                <div className="result-card result-benefit">
                  <span>Estimated net benefit</span>
                  <strong>{formatMoney(netBenefit)}</strong>
                  <small>Synthetic held-out backtest</small>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      <p className="simulation-disclosure">
        Results are estimates from synthetic held-out data. They are not
        guaranteed financial outcomes.
      </p>
    </section>
  );
}