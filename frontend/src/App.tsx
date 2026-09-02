import { useEffect, useState } from "react";
import "./App.css";

type HealthResponse = {
  status: string;
  service: string;
  version: string;
};

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/health")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Backend request failed");
        }

        return response.json();
      })
      .then((data: HealthResponse) => setHealth(data))
      .catch(() => setError("Could not connect to the backend"));
  }, []);

  return (
    <main className="page">
      <section className="card">
        <p className="label">RAZORPAY AI BUILDATHON 2026</p>
        <h1>Chargeback Radar</h1>

        <p className="description">
          Predict chargeback risk and recommend the safest defensive action.
        </p>

        {health && (
          <div className="status success">
            <span className="dot" />
            Backend connected: {health.status}
          </div>
        )}

        {!health && !error && (
          <div className="status">Checking backend connection...</div>
        )}

        {error && <div className="status error">{error}</div>}
      </section>
    </main>
  );
}

export default App;