import { useEffect, useMemo, useState } from "react";

type Transaction = {
  payment_id: string;
  customer_id: string;
  created_at: string;
  amount_paise: number;
  calibrated_probability: number;
  recommended_action: string;
  target?: number;
  chargeback_within_120d?: number;
};

type ApiResponse =
  | Transaction[]
  | {
      count: number;
      items: Transaction[];
    };

const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function formatMoney(paise: number) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(paise / 100);
}

function formatAction(action: string) {
  return action
    .toLowerCase()
    .split("_")
    .map((word) => word[0].toUpperCase() + word.slice(1))
    .join(" ");
}

export default function TransactionQueue() {
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [selectedAction, setSelectedAction] = useState("ALL");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadTransactions() {
      try {
        setLoading(true);

        const response = await fetch(
          `${API_URL}/api/transactions?limit=50`
        );

        if (!response.ok) {
          throw new Error(`API returned ${response.status}`);
        }

        const data: ApiResponse = await response.json();

        setTransactions(Array.isArray(data) ? data : data.items);
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Could not load transactions"
        );
      } finally {
        setLoading(false);
      }
    }

    loadTransactions();
  }, []);

  const filteredTransactions = useMemo(() => {
    if (selectedAction === "ALL") {
      return transactions;
    }

    return transactions.filter(
      (transaction) =>
        transaction.recommended_action === selectedAction
    );
  }, [transactions, selectedAction]);

  if (loading) {
    return <section className="queue-panel">Loading transaction queue…</section>;
  }

  if (error) {
    return (
      <section className="queue-panel queue-error">
        Could not load transactions: {error}
      </section>
    );
  }

  return (
    <section className="queue-panel">
      <div className="queue-header">
        <div>
          <h2>High-risk transaction queue</h2>
          <p>
            Highest calibrated risks from the held-out evaluation period
          </p>
        </div>

        <select
          value={selectedAction}
          onChange={(event) => setSelectedAction(event.target.value)}
          aria-label="Filter by recommended action"
        >
          <option value="ALL">All actions</option>
          <option value="PREPARE_EVIDENCE">Prepare evidence</option>
          <option value="MANUAL_REVIEW">Manual review</option>
          <option value="RECOMMEND_REFUND">Recommend refund</option>
          <option value="MONITOR">Monitor</option>
        </select>
      </div>

      <div className="table-wrapper">
        <table className="transaction-table">
          <thead>
            <tr>
              <th>Payment</th>
              <th>Date</th>
              <th>Amount</th>
              <th>Risk</th>
              <th>Recommended action</th>
              <th>Actual outcome</th>
            </tr>
          </thead>

          <tbody>
            {filteredTransactions.map((transaction) => {
              const outcome =
                transaction.chargeback_within_120d ??
                transaction.target ??
                0;

              const risk = transaction.calibrated_probability * 100;

              return (
                <tr key={transaction.payment_id}>
                  <td>
                    <strong>{transaction.payment_id}</strong>
                    <span>{transaction.customer_id}</span>
                  </td>

                  <td>
                    {new Date(transaction.created_at).toLocaleDateString(
                      "en-IN",
                      {
                        day: "2-digit",
                        month: "short",
                        year: "numeric",
                      }
                    )}
                  </td>

                  <td>{formatMoney(transaction.amount_paise)}</td>

                  <td>
                    <div className="risk-cell">
                      <strong>{risk.toFixed(2)}%</strong>
                      <div className="risk-track">
                        <div
                          className="risk-fill"
                          style={{
                            width: `${Math.min(risk * 4, 100)}%`,
                          }}
                        />
                      </div>
                    </div>
                  </td>

                  <td>
                    <span
                      className={`action-badge action-${transaction.recommended_action.toLowerCase()}`}
                    >
                      {formatAction(transaction.recommended_action)}
                    </span>
                  </td>

                  <td>
                    <span
                      className={
                        outcome === 1
                          ? "outcome-positive"
                          : "outcome-negative"
                      }
                    >
                      {outcome === 1 ? "Chargeback" : "No chargeback"}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}