import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

type FrontierPoint = {
  manual_review_cost_rupees: number;
  intervention_count: number;
  intervention_rate: number;
  precision: number;
  recall: number;
  true_positive: number;
  false_positive: number;
  false_negative: number;
  policy_cost_rupees: number;
  false_positive_cost_rupees: number;
  estimated_net_benefit_rupees: number;
  estimated_benefit_rate: number;
  action_mix: Record<string, number>;
};

type FrontierResponse = {
  records_evaluated: number;
  variable: string;
  default_scenario: FrontierPoint;
  highest_estimated_benefit_scenario: FrontierPoint;
  points: FrontierPoint[];
  disclosures: string[];
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

export default function PolicyFrontier() {
  const [frontier, setFrontier] =
    useState<FrontierResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadFrontier() {
      try {
        setLoading(true);
        setError("");

        const response = await fetch(
          `${API_URL}/api/policy/frontier`,
        );

        if (!response.ok) {
          throw new Error(
            `Policy frontier failed with ${response.status}`,
          );
        }

        const data: FrontierResponse = await response.json();
        setFrontier(data);
      } catch (requestError) {
        setError(
          requestError instanceof Error
            ? requestError.message
            : "Could not load policy frontier.",
        );
      } finally {
        setLoading(false);
      }
    }

    void loadFrontier();
  }, []);

  if (loading) {
    return (
      <section className="frontier-panel">
        <p>Loading policy frontier…</p>
      </section>
    );
  }

  if (error || !frontier) {
    return (
      <section className="frontier-panel">
        <p className="simulation-error">
          {error || "Policy frontier unavailable."}
        </p>
      </section>
    );
  }

  const defaultScenario = frontier.default_scenario;
  const highestBenefit =
    frontier.highest_estimated_benefit_scenario;

  return (
    <section className="frontier-panel">
      <div className="frontier-heading">
        <div>
          <span className="section-kicker">
            Held-out sensitivity analysis
          </span>
          <h2>Policy cost frontier</h2>
          <p>
            See how review cost changes intervention volume,
            chargeback coverage and estimated merchant benefit.
          </p>
        </div>

        <div className="frontier-badge">
          {frontier.records_evaluated.toLocaleString("en-IN")} test
          payments
        </div>
      </div>

      <div className="frontier-summary">
        <div>
          <span>Default review cost</span>
          <strong>
            {formatMoney(
              defaultScenario.manual_review_cost_rupees,
            )}
          </strong>
        </div>

        <div>
          <span>Default net benefit</span>
          <strong className="positive-value">
            {formatMoney(
              defaultScenario.estimated_net_benefit_rupees,
            )}
          </strong>
        </div>

        <div>
          <span>Default false positives</span>
          <strong>
            {defaultScenario.false_positive.toLocaleString("en-IN")}
          </strong>
        </div>

        <div>
          <span>Best scenario in grid</span>
          <strong>
            {formatMoney(
              highestBenefit.manual_review_cost_rupees,
            )}{" "}
            review
          </strong>
        </div>
      </div>

      <div className="frontier-charts">
        <div className="frontier-chart-card">
          <h3>Economics</h3>
          <p>Estimated benefit and false-positive cost.</p>

          <div className="frontier-chart">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={frontier.points}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="#e2e8f0"
                />
                <XAxis
                  dataKey="manual_review_cost_rupees"
                  tickFormatter={(value) => `₹${Number(value)}`}
                />
                <YAxis
                  tickFormatter={(value) =>
                    `₹${Math.round(Number(value) / 1000)}k`
                  }
                />
                <Tooltip
                  labelFormatter={(value) =>
                    `Review cost: ${formatMoney(Number(value))}`
                  }
                  formatter={(value, name) => [
                    formatMoney(Number(value)),
                    String(name),
                  ]}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="estimated_net_benefit_rupees"
                  name="Net benefit"
                  stroke="#059669"
                  strokeWidth={3}
                  dot={{ r: 3 }}
                />
                <Line
                  type="monotone"
                  dataKey="false_positive_cost_rupees"
                  name="False-positive cost"
                  stroke="#dc2626"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="frontier-chart-card">
          <h3>Risk coverage</h3>
          <p>Precision, recall and total intervention rate.</p>

          <div className="frontier-chart">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={frontier.points}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="#e2e8f0"
                />
                <XAxis
                  dataKey="manual_review_cost_rupees"
                  tickFormatter={(value) => `₹${Number(value)}`}
                />
                <YAxis
                  domain={[0, 1]}
                  tickFormatter={(value) =>
                    `${Math.round(Number(value) * 100)}%`
                  }
                />
                <Tooltip
                  labelFormatter={(value) =>
                    `Review cost: ${formatMoney(Number(value))}`
                  }
                  formatter={(value, name) => [
                    formatPercent(Number(value)),
                    String(name),
                  ]}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="precision"
                  name="Precision"
                  stroke="#7c3aed"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
                <Line
                  type="monotone"
                  dataKey="recall"
                  name="Recall"
                  stroke="#2563eb"
                  strokeWidth={3}
                  dot={{ r: 3 }}
                />
                <Line
                  type="monotone"
                  dataKey="intervention_rate"
                  name="Intervention rate"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <p className="simulation-disclosure">
        The highest-benefit point is sensitivity analysis, not a
        guaranteed recommendation. All scenarios use the same untouched
        customer-disjoint synthetic test population.
      </p>
    </section>
  );
}
