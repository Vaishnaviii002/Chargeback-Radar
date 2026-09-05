import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useState,
} from "react";
import {
  Activity,
  Database,
  IndianRupee,
  RefreshCw,
  ShieldCheck,
  Target,
} from "lucide-react";
import TransactionQueue from "./components/TransactionQueue";
import PolicySimulator from "./components/PolicySimulator";
import { fetchMetrics, type MetricsResponse } from "./api";
import EffectivenessSensitivity from "./components/EffectivenessSensitivity";
import "./App.css";
import EvidenceWorkbench from "./components/EvidenceWorkbench";
import ModelExplanationWorkbench from "./components/ModelExplanationWorkbench";
import "./components/ModelExplanationWorkbench.css";
import RazorpayTestCheckout from "./components/RazorpayTestCheckout";
import "./components/RazorpayTestCheckout.css";

const PolicyFrontier = lazy(
  () => import("./components/PolicyFrontier"),
);


const ACTIONS = [
  {
    key: "MONITOR",
    label: "Monitor",
    color: "#64748b",
  },
  {
    key: "PREPARE_EVIDENCE",
    label: "Prepare evidence",
    color: "#7c3aed",
  },
  {
    key: "MANUAL_REVIEW",
    label: "Manual review",
    color: "#f59e0b",
  },
  {
    key: "RECOMMEND_REFUND",
    label: "Recommend refund",
    color: "#ef4444",
  },
];

function formatPercent(value: number, digits = 1) {
  return new Intl.NumberFormat("en-IN", {
    style: "percent",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

function formatNumber(value: number) {
  return new Intl.NumberFormat("en-IN").format(value);
}

function formatRupees(value: number) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(value);
}

function App() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);

  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const loadMetrics = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetchMetrics();
      setMetrics(response);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Could not load dashboard metrics.",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    void fetchMetrics()
      .then((response) => {
        if (!cancelled) {
          setMetrics(response);
        }
      })
      .catch((requestError: unknown) => {
        if (!cancelled) {
          setError(
            requestError instanceof Error
              ? requestError.message
              : "Could not load dashboard metrics.",
          );
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) {
    return (
      <main className="state-screen">
        <div className="state-spinner" />
        <h1>Loading Chargeback Radar</h1>
        <p>Reading held-out model results…</p>
      </main>
    );
  }

  if (error || !metrics) {
    return (
      <main className="state-screen">
        <div className="error-symbol">!</div>
        <h1>Backend connection failed</h1>
        <p>{error}</p>

        <button className="primary-button" onClick={() => void loadMetrics()}>
          <RefreshCw size={16} />
          Try again
        </button>
      </main>
    );
  }

  const evaluation = metrics.evaluation;
  const policy = metrics.default_policy;
  const performance = evaluation.model_performance;
  const confusion = evaluation.confusion_matrix;

  const cards = [
    {
      label: "Held-out payments",
      value: formatNumber(evaluation.dataset.test_rows),
      detail: `${formatNumber(
        evaluation.dataset.positive_labels,
      )} actual chargebacks`,
      icon: Database,
    },
    {
      label: "Average precision",
      value: formatPercent(performance.average_precision, 2),
      detail: `${formatPercent(
        evaluation.dataset.base_rate,
        2,
      )} random baseline`,
      icon: Activity,
    },
    {
      label: "Precision",
      value: formatPercent(performance.precision),
      detail: "Of payments flagged",
      icon: Target,
    },
    {
      label: "Recall",
      value: formatPercent(performance.recall),
      detail: "Of chargebacks detected",
      icon: ShieldCheck,
    },
    {
      label: "False-positive cost",
      value: formatRupees(evaluation.false_positive_cost.total_rupees),
      detail: `${formatNumber(
        evaluation.false_positive_cost.false_positive_count,
      )} unnecessary reviews`,
      icon: IndianRupee,
    },
    {
      label: "Estimated net benefit",
      value: formatRupees(policy.costs.estimated_net_benefit_rupees),
      detail: "Under displayed assumptions",
      icon: IndianRupee,
      positive: policy.costs.estimated_net_benefit_rupees >= 0,
    },
  ];

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <ShieldCheck size={22} />
          </div>

          <div>
            <strong>Chargeback Radar</strong>
            <span>AI Risk Manager</span>
          </div>
        </div>

        <div className="topbar-status">
          <span className="status-dot" />
          Defense-only system
        </div>
      </header>

      <main className="dashboard">
        <section className="page-heading">
          <div>
            <p className="eyebrow">Razorpay AI Build Hackathon · Track 02</p>

            <h1>Risk command center</h1>

            <p className="heading-copy">
              Detect likely card chargebacks and choose the lowest-cost
              defensive action.
            </p>
          </div>

          <button
            className="secondary-button"
            onClick={() => void loadMetrics()}
          >
            <RefreshCw size={16} />
            Refresh data
          </button>
        </section>

        <section className="metric-grid">
          {cards.map((card) => {
            const Icon = card.icon;

            return (
              <article className="metric-card" key={card.label}>
                <div className="metric-label">
                  <span>{card.label}</span>
                  <Icon size={17} />
                </div>

                <strong
                  className={
                    card.positive ? "metric-value positive" : "metric-value"
                  }
                >
                  {card.value}
                </strong>

                <span className="metric-detail">{card.detail}</span>
              </article>
            );
          })}
        </section>

        <section className="analysis-grid">
          <article className="panel policy-panel">
            <div className="panel-heading">
              <div>
                <h2>Default policy allocation</h2>
                <p>Recommended actions across the held-out test population.</p>
              </div>

              <span className="panel-stat">
                {formatPercent(policy.intervention_rate)} intervened
              </span>
            </div>

            <div className="action-list">
              {ACTIONS.map((action) => {
                const count = policy.action_mix[action.key] ?? 0;

                const share =
                  policy.records_evaluated > 0
                    ? count / policy.records_evaluated
                    : 0;

                return (
                  <div className="action-row" key={action.key}>
                    <div className="action-meta">
                      <span>
                        <i
                          style={{
                            background: action.color,
                          }}
                        />
                        {action.label}
                      </span>

                      <strong>
                        {formatNumber(count)}
                        <small>{formatPercent(share)}</small>
                      </strong>
                    </div>

                    <div className="progress-track">
                      <div
                        className="progress-value"
                        style={{
                          width: `${Math.max(
                            share * 100,
                            share > 0 ? 0.8 : 0,
                          )}%`,
                          background: action.color,
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </article>

          <article className="panel">
            <div className="panel-heading">
              <div>
                <h2>Held-out confusion matrix</h2>
                <p>Exact outcomes at the fixed operating threshold.</p>
              </div>
            </div>

            <div className="confusion-grid">
              <div className="confusion-cell correct">
                <span>True positive</span>
                <strong>{formatNumber(confusion.true_positive)}</strong>
                <small>Correctly flagged</small>
              </div>

              <div className="confusion-cell warning">
                <span>False positive</span>
                <strong>{formatNumber(confusion.false_positive)}</strong>
                <small>Unnecessary review</small>
              </div>

              <div className="confusion-cell danger">
                <span>False negative</span>
                <strong>{formatNumber(confusion.false_negative)}</strong>
                <small>Chargeback missed</small>
              </div>

              <div className="confusion-cell neutral">
                <span>True negative</span>
                <strong>{formatNumber(confusion.true_negative)}</strong>
                <small>Correctly monitored</small>
              </div>
            </div>
          </article>
        </section>

        <section className="method-strip">
          <ShieldCheck size={19} />

          <div>
            <strong>Honest evaluation</strong>
            <span>
              {evaluation.disclosures.split}. Probabilities calibrated using{" "}
              {metrics.calibration.selected_method}.{" "}
              {evaluation.disclosures.money_note}
            </span>
          </div>
        </section>
        <RazorpayTestCheckout />
        <ModelExplanationWorkbench />
        <EvidenceWorkbench />
        <PolicySimulator />
        <EffectivenessSensitivity />
        <Suspense
          fallback={
            <section className="panel" aria-busy="true">
              Loading policy frontier…
            </section>
          }
        >
          <PolicyFrontier />
        </Suspense>
        <TransactionQueue />
      </main>
    </div>
  );
}

export default App;
