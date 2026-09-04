import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Bot,
  CheckCircle2,
  Clock3,
  FileCheck2,
  Fingerprint,
  LockKeyhole,
  RefreshCw,
  SearchCheck,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import "./EvidenceWorkbench.css";

import { fetchTransactions, type Transaction } from "../api";

import {
  fetchEvidenceStatus,
  generateTransactionEvidence,
  verifyEvidenceAudit,
  type AuditVerification,
  type EvidenceDeliveryResult,
  type EvidenceSystemStatus,
} from "../evidenceApi";

function formatAction(value: string) {
  return value
    .toLowerCase()
    .split("_")
    .map((word) => word[0].toUpperCase() + word.slice(1))
    .join(" ");
}

function formatRupees(paise: number) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(paise / 100);
}

function deliveryLabel(result: EvidenceDeliveryResult) {
  if (result.delivery_mode === "LIVE_OPENAI") {
    return "Live OpenAI draft";
  }

  if (result.delivery_mode === "CACHE") {
    return result.fallback_used
      ? "Cached deterministic fallback"
      : "Validated cache";
  }

  return "Deterministic fallback";
}

export default function EvidenceWorkbench() {
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [selectedPaymentId, setSelectedPaymentId] = useState("");
  const [systemStatus, setSystemStatus] =
    useState<EvidenceSystemStatus | null>(null);
  const [result, setResult] =
    useState<EvidenceDeliveryResult | null>(null);
  const [audit, setAudit] = useState<AuditVerification | null>(null);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [checkingAudit, setCheckingAudit] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadWorkbench() {
      try {
        setLoading(true);
        setError(null);

        const [transactionResponse, statusResponse] = await Promise.all([
          fetchTransactions(50),
          fetchEvidenceStatus(),
        ]);

        setTransactions(transactionResponse.transactions);
        setSystemStatus(statusResponse);

        const preferred =
          transactionResponse.transactions.find(
            (transaction) =>
              transaction.recommended_action !== "MONITOR",
          ) ?? transactionResponse.transactions[0];

        setSelectedPaymentId(preferred?.payment_id ?? "");
      } catch (requestError) {
        setError(
          requestError instanceof Error
            ? requestError.message
            : "Could not load the evidence workbench.",
        );
      } finally {
        setLoading(false);
      }
    }

    void loadWorkbench();
  }, []);

  const selectedTransaction = useMemo(
    () =>
      transactions.find(
        (transaction) =>
          transaction.payment_id === selectedPaymentId,
      ) ?? null,
    [selectedPaymentId, transactions],
  );

  function choosePayment(paymentId: string) {
    setSelectedPaymentId(paymentId);
    setResult(null);
    setAudit(null);
    setError(null);
  }

  async function generate(refresh: boolean) {
    if (!selectedPaymentId) {
      return;
    }

    try {
      setGenerating(true);
      setError(null);
      setAudit(null);

      const response = await generateTransactionEvidence(
        selectedPaymentId,
        refresh,
      );

      setResult(response);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Evidence generation failed safely.",
      );
    } finally {
      setGenerating(false);
    }
  }

  async function checkAudit() {
    try {
      setCheckingAudit(true);
      setError(null);
      setAudit(await verifyEvidenceAudit());
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Audit verification failed.",
      );
    } finally {
      setCheckingAudit(false);
    }
  }

  return (
    <section className="evidence-workbench">
      <div className="evidence-heading">
        <div>
          <span className="section-kicker">AI Evidence Copilot</span>

          <h2>Chargeback evidence workbench</h2>

          <p>
            Convert sanitized, timestamped case facts into a cited analyst
            draft. The model explains the deterministic policy; it never
            executes an action.
          </p>
        </div>

        <div className="evidence-system-badges">
          <span
            className={
              systemStatus?.api_key_configured
                ? "evidence-badge success"
                : "evidence-badge warning"
            }
          >
            <Bot size={14} />

            {systemStatus?.api_key_configured
              ? `${systemStatus.model} ready`
              : "Fallback mode ready"}
          </span>

          <span className="evidence-badge neutral">
            <LockKeyhole size={14} />
            No automatic actions
          </span>
        </div>
      </div>

      {loading ? (
        <div className="evidence-loading">
          <RefreshCw className="spin-icon" size={18} />
          Loading evidence controls…
        </div>
      ) : (
        <div className="evidence-layout">
          <aside className="evidence-case-picker">
            <label htmlFor="evidence-payment">Held-out payment</label>

            <select
              id="evidence-payment"
              value={selectedPaymentId}
              onChange={(event) =>
                choosePayment(event.target.value)
              }
            >
              {transactions.map((transaction) => (
                <option
                  key={transaction.payment_id}
                  value={transaction.payment_id}
                >
                  {transaction.payment_id} ·{" "}
                  {formatAction(transaction.recommended_action)}
                </option>
              ))}
            </select>

            {selectedTransaction && (
              <div className="evidence-case-summary">
                <div>
                  <span>Amount</span>

                  <strong>
                    {formatRupees(selectedTransaction.amount_paise)}
                  </strong>
                </div>

                <div>
                  <span>Calibrated risk</span>

                  <strong>
                    {(
                      selectedTransaction.calibrated_probability *
                      100
                    ).toFixed(2)}
                    %
                  </strong>
                </div>

                <div className="evidence-case-action">
                  <span>Deterministic policy</span>

                  <strong>
                    {formatAction(
                      selectedTransaction.recommended_action,
                    )}
                  </strong>
                </div>
              </div>
            )}

            <button
              className="evidence-generate-button"
              disabled={!selectedPaymentId || generating}
              onClick={() => void generate(false)}
            >
              {generating ? (
                <RefreshCw className="spin-icon" size={16} />
              ) : (
                <Sparkles size={16} />
              )}

              {generating ? "Generating…" : "Generate evidence"}
            </button>

            <button
              className="evidence-refresh-button"
              disabled={!selectedPaymentId || generating}
              onClick={() => void generate(true)}
            >
              <RefreshCw size={15} />
              Refresh live draft
            </button>

            <p className="evidence-picker-note">
              Refresh bypasses cache. If the provider is unavailable,
              the same request returns a grounded deterministic fallback.
            </p>
          </aside>

          <div className="evidence-result-area">
            {error && (
              <div className="evidence-error" role="alert">
                <AlertTriangle size={18} />

                <div>
                  <strong>Request stopped safely</strong>
                  <span>{error}</span>
                </div>
              </div>
            )}

            {!result && !generating && (
              <div className="evidence-empty">
                <FileCheck2 size={30} />

                <h3>Ready to build a cited evidence pack</h3>

                <p>
                  Choose a payment and generate an analyst-ready draft
                  from allowlisted facts. Outcome labels and customer
                  identifiers are excluded before the model boundary.
                </p>
              </div>
            )}

            {generating && (
              <div className="evidence-empty">
                <RefreshCw className="spin-icon" size={28} />

                <h3>Building and validating the draft</h3>

                <p>
                  Compiling facts, applying output guardrails and
                  recording an integrity-protected audit event.
                </p>
              </div>
            )}

            {result && !generating && (
              <div className="evidence-result">
                <div className="evidence-provenance">
                  <span
                    className={`delivery-mode ${result.delivery_mode.toLowerCase()}`}
                  >
                    <Sparkles size={14} />
                    {deliveryLabel(result)}
                  </span>

                  <span>
                    <SearchCheck size={14} />
                    {result.guardrails.checks_run} guardrails passed
                  </span>

                  <span>
                    <Fingerprint size={14} />
                    {result.guardrails.citations_checked} citations checked
                  </span>

                  <span>
                    <Clock3 size={14} />
                    {result.latency_ms} ms
                  </span>
                </div>

                <div className="evidence-result-heading">
                  <div>
                    <span>Case strength</span>
                    <strong>
                      {result.evidence_pack.case_strength}
                    </strong>
                  </div>

                  <div>
                    <span>Recommended action</span>

                    <strong>
                      {formatAction(
                        result.evidence_pack.recommended_action,
                      )}
                    </strong>
                  </div>
                </div>

                <article className="evidence-narrative">
                  <h3>Case summary</h3>
                  <p>{result.evidence_pack.case_summary}</p>

                  <h3>Action rationale</h3>
                  <p>{result.evidence_pack.action_rationale}</p>
                </article>

                <div className="evidence-section-heading">
                  <div>
                    <h3>Cited evidence</h3>
                    <p>
                      Every statement maps to an allowlisted fact ID.
                    </p>
                  </div>

                  <span>
                    {result.evidence_pack.evidence_items.length} items
                  </span>
                </div>

                <div className="evidence-item-list">
                  {result.evidence_pack.evidence_items.map(
                    (item, index) => (
                      <article
                        className="evidence-item"
                        key={`${item.title}-${index}`}
                      >
                        <div className="evidence-item-heading">
                          <h4>{item.title}</h4>

                          <span
                            className={`strength-${item.strength.toLowerCase()}`}
                          >
                            {item.strength}
                          </span>
                        </div>

                        <p>{item.statement}</p>
                        <small>{item.relevance}</small>

                        <div className="evidence-citations">
                          {item.citation_fact_ids.map((factId) => (
                            <code key={factId}>{factId}</code>
                          ))}
                        </div>
                      </article>
                    ),
                  )}
                </div>

                {result.evidence_pack.missing_evidence.length > 0 && (
                  <>
                    <div className="evidence-section-heading missing-heading">
                      <div>
                        <h3>Missing evidence</h3>

                        <p>
                          Records an analyst should collect before action.
                        </p>
                      </div>
                    </div>

                    <div className="missing-evidence-list">
                      {result.evidence_pack.missing_evidence.map(
                        (item) => (
                          <article key={item.item}>
                            <div>
                              <AlertTriangle size={15} />
                              <strong>{item.item}</strong>
                              <span>{item.priority}</span>
                            </div>

                            <p>{item.reason}</p>

                            <small>
                              Expected source: {item.expected_source}
                            </small>
                          </article>
                        ),
                      )}
                    </div>
                  </>
                )}

                {result.evidence_pack.customer_message && (
                  <article className="customer-draft">
                    <div>
                      <h3>Customer message</h3>
                      <span>DRAFT ONLY</span>
                    </div>

                    <strong>
                      {
                        result.evidence_pack.customer_message
                          .subject
                      }
                    </strong>

                    <p>
                      {result.evidence_pack.customer_message.body}
                    </p>
                  </article>
                )}

                <div className="evidence-limitations">
                  <h3>Limitations</h3>

                  <ul>
                    {result.evidence_pack.limitations.map(
                      (limitation) => (
                        <li key={limitation}>{limitation}</li>
                      ),
                    )}
                  </ul>
                </div>

                <div className="evidence-safety-footer">
                  <div>
                    <ShieldCheck size={19} />

                    <span>
                      <strong>Human approval required</strong>
                      No action has been executed
                    </span>
                  </div>

                  <button
                    disabled={checkingAudit}
                    onClick={() => void checkAudit()}
                  >
                    {checkingAudit ? (
                      <RefreshCw
                        className="spin-icon"
                        size={15}
                      />
                    ) : (
                      <Fingerprint size={15} />
                    )}

                    Verify audit chain
                  </button>
                </div>

                {audit && (
                  <div className="audit-verification">
                    <CheckCircle2 size={17} />

                    <div>
                      <strong>Audit chain verified</strong>

                      <span>
                        {audit.records_verified} records ·{" "}
                        {audit.integrity_mode}
                        {" · head "}
                        {audit.last_event_hash.slice(0, 12)}…
                      </span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  );
}