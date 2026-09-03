import { useEffect, useState } from "react";


const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

type ScenarioCosts = {
  do_nothing_baseline_rupees: number;
  remaining_chargeback_loss_rupees: number;
  gross_avoided_loss_rupees: number;
  intervention_cost_rupees: number;
  policy_cost_rupees: number;
  false_positive_cost_rupees: number;
  estimated_net_benefit_rupees: number;
};

type EffectivenessScenario = {
  scenario: "LOW" | "BASE" | "HIGH";
  assumptions: {
    evidence_recovery_rate: number;
    review_prevention_rate: number;
  };
  intervention_count: number;
  intervention_rate: number;
  precision: number;
  recall: number;
  action_mix: Record<string, number>;
  costs: ScenarioCosts;
};

type SensitivityResponse = {
  records_evaluated: number;
  scenario_order: string[];
  scenarios: EffectivenessScenario[];
  ranges: {
    estimated_net_benefit_rupees: {
      minimum: number;
      maximum: number;
    };
  };
  disclosure: string;
};

function formatMoney(value: number) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(value);
}

function formatPercent(value: number) {
  return new Intl.NumberFormat("en-IN", {
    style: "percent",
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  }).format(value);
}

function scenarioLabel(scenario: string) {
  if (scenario === "LOW") return "Conservative";
  if (scenario === "HIGH") return "Optimistic";
  return "Expected";
}

export default function EffectivenessSensitivity() {
  const [report, setReport] =
    useState<SensitivityResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadReport() {
      try {
        setLoading(true);
        setError("");

        const response = await fetch(
          `${API_URL}/api/policy/effectiveness`,
        );

        if (!response.ok) {
          throw new Error(
            `Effectiveness analysis failed with ${response.status}`,
          );
        }

        const data: SensitivityResponse = await response.json();
        setReport(data);
      } catch (requestError) {
        setError(
          requestError instanceof Error
            ? requestError.message
            : "Could not load effectiveness analysis.",
        );
      } finally {
        setLoading(false);
      }
    }

    void loadReport();
  }, []);

  if (loading) {
    return (
      <section className="sensitivity-panel">
        <p>Loading effectiveness scenarios…</p>
      </section>
    );
  }

  if (error || !report) {
    return (
      <section className="sensitivity-panel">
        <p className="simulation-error">
          {error || "Effectiveness analysis unavailable."}
        </p>
      </section>
    );
  }

  const benefitRange =
    report.ranges.estimated_net_benefit_rupees;
  const conservativeIsPositive = benefitRange.minimum > 0;

  return (
    <section className="sensitivity-panel">
      <div className="sensitivity-heading">
        <div>
          <span className="section-kicker">
            Assumption stress test
          </span>
          <h2>Intervention effectiveness sensitivity</h2>
          <p>
            The same held-out payments are replayed under weaker and
            stronger intervention-effectiveness assumptions.
          </p>
        </div>

        <div
          className={
            conservativeIsPositive
              ? "sensitivity-verdict positive"
              : "sensitivity-verdict caution"
          }
        >
          <span>Observed benefit range</span>
          <strong>
            {formatMoney(benefitRange.minimum)} –{" "}
            {formatMoney(benefitRange.maximum)}
          </strong>
          <small>
            {conservativeIsPositive
              ? "Positive in every tested scenario"
              : "Not positive in every tested scenario"}
          </small>
        </div>
      </div>

      <div className="sensitivity-grid">
        {report.scenarios.map((scenario) => (
          <article
            className={
              scenario.scenario === "BASE"
                ? "sensitivity-card base"
                : "sensitivity-card"
            }
            key={scenario.scenario}
          >
            <div className="sensitivity-card-heading">
              <div>
                <span>{scenario.scenario}</span>
                <h3>{scenarioLabel(scenario.scenario)}</h3>
              </div>

              {scenario.scenario === "BASE" && (
                <i>Displayed default</i>
              )}
            </div>

            <div className="assumption-row">
              <span>Evidence recovery</span>
              <strong>
                {formatPercent(
                  scenario.assumptions.evidence_recovery_rate,
                )}
              </strong>
            </div>

            <div className="assumption-row">
              <span>Review prevention</span>
              <strong>
                {formatPercent(
                  scenario.assumptions.review_prevention_rate,
                )}
              </strong>
            </div>

            <div className="sensitivity-metric-grid">
              <div>
                <span>Recall</span>
                <strong>{formatPercent(scenario.recall)}</strong>
              </div>
              <div>
                <span>Interventions</span>
                <strong>
                  {scenario.intervention_count.toLocaleString("en-IN")}
                </strong>
              </div>
              <div>
                <span>Gross avoided</span>
                <strong>
                  {formatMoney(
                    scenario.costs.gross_avoided_loss_rupees,
                  )}
                </strong>
              </div>
              <div>
                <span>Intervention cost</span>
                <strong>
                  {formatMoney(
                    scenario.costs.intervention_cost_rupees,
                  )}
                </strong>
              </div>
            </div>

            <div className="scenario-benefit">
              <span>Estimated net benefit</span>
              <strong
                className={
                  scenario.costs.estimated_net_benefit_rupees >= 0
                    ? "positive-value"
                    : "negative-value"
                }
              >
                {formatMoney(
                  scenario.costs.estimated_net_benefit_rupees,
                )}
              </strong>
            </div>
          </article>
        ))}
      </div>

      <p className="simulation-disclosure">
        {report.disclosure} These values are backtest estimates, not
        guaranteed savings or statistical confidence intervals.
      </p>
    </section>
  );
}