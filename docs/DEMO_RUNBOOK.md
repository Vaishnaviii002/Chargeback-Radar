# Five-Minute Demo Runbook

This runbook is for Phase 10 recording. It uses synthetic held-out reports and Razorpay Test Mode only.

## Prerequisites

- Complete `.\scripts\verify_release.ps1` successfully before recording.
- Install backend/frontend dependencies and generate baseline artifacts as described in `README.md`.
- Copy `.env.example` to ignored `.env`.
- For Checkout only, add Razorpay Test Mode key ID/secret and set `RAZORPAY_INTEGRATION_ENABLED=true`, `RAZORPAY_TEST_MODE=true`.
- For the local webhook replay segment, add a Test Mode webhook secret and set `RAZORPAY_WEBHOOK_ENABLED=true`.
- Do not put secrets, card details, signatures, or a temporary tunnel URL on screen.

Obtain current Test Mode card details from the [official Razorpay Test Card Details page](https://razorpay.com/docs/payments/payments/test-card-details/?preferred-country=IN). Do not record or commit them in this repository. Razorpay states that its test cards are for Test Mode and no real money is deducted.

## Start and verify

Backend terminal:

```powershell
Set-Location C:\Users\hp\desktop\chargeback-radar
.\.venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Frontend terminal:

```powershell
Set-Location C:\Users\hp\desktop\chargeback-radar
npm --prefix frontend run dev -- --host 127.0.0.1
```

Health-check terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

Open `http://127.0.0.1:5173`. Keep API docs at `http://127.0.0.1:8000/docs` in a second tab only if needed.

## Recommended screen order

### 0:00–0:35 — Product and evaluation boundary

Show the product header and say: “Chargeback Radar is a defense-only AI Risk Manager. It estimates the synthetic target `chargeback_within_120d`, calibrates the probability, and recommends a bounded merchant response. It does not adjudicate fraud or execute money movement.”

Point to the held-out payment count and the months 1–9 / month 10 / months 11–12, customer-disjoint design. State that all data is synthetic and performance does not establish real-world effectiveness.

### 0:35–1:15 — Held-out metrics and calibration

Show calibrated AP 14.926%, Brier 0.006258, precision 19.466%, recall 58.621%, and the 0.715% base rate. Explain that isotonic calibration converts model scores into more decision-useful probabilities. Do not call this accuracy and do not claim production validation.

Show the 2% model review-capacity policy and the visible cost assumptions. Explain that deterministic merchant-error overrides can raise total policy intervention beyond model review capacity.

### 1:15–2:00 — Queue and SHAP

Open a high-risk held-out payment from the queue. Show the calibrated probability, risk band, and advisory action. Scroll to its increasing and decreasing factors and read the exact label:

> Model explanation — not evidence.

Explain that SHAP factors contribute to the model estimate but do not establish causation, fraud, customer intent, or dispute facts.

### 2:00–2:45 — Evidence Copilot and fallback

Generate the evidence draft for the same payment. Point to exact fact IDs, missing evidence, the draft-only customer message (if present), human approval, and `action_executed=false`. Explain that model explanation and evidence are separate.

For the deterministic fallback, keep `EVIDENCE_AI_ENABLED=false` before starting the backend and show the `DETERMINISTIC_FALLBACK` delivery mode. State that optional AI can phrase verified facts but cannot add facts, citations, or authority.

### 2:45–3:20 — Timestamp-safe support ablation

Name the three signals: `intent_to_cancel`, `non_receipt_complaint`, and `dissatisfaction`. State that 8,270 events were observable and 5,297 future events were excluded; raw text, PII, outcomes, and prompt instructions do not enter the report.

Present the complete trade-off: AP improved by about 0.003508 and recall by 4.598 percentage points; precision decreased by about 0.099 percentage points, while Brier improved by about 0.000058. Do not imply causation or real-world lift.

### 3:20–4:25 — Razorpay Test Mode Checkout

Point to both “Razorpay Test Mode” and “No real money.” Choose the normal or heightened merchant-declared context profile and create the order. Complete card Checkout using the official Test Mode details off-screen where practical.

Show the backend result: Checkout signature verified, provider order/payment binding passed, payment status is authorized or captured, model version is `0.1.0`, calibration is isotonic, held-out evaluation is unaffected, human approval is required, and `financial_action_executed=false`.

Say clearly that merchant context is declared capture-time context, not Razorpay-supplied customer intelligence.

### 4:25–4:50 — Webhook signature and replay

With the backend running and the ignored local `.env` configured, run:

```powershell
.\.venv\Scripts\python.exe scripts/demo_webhook.py
```

Show `PROCESSED` for the signed first request and `IDEMPOTENT_REPLAY` for the exact repeat, plus human approval true and financial action false. The helper never prints the secret or signature and sends no PII. Explain that exact raw bytes are verified before parsing and only metadata is stored.

If discussing the prior public test, say: “A locally signed request reached the endpoint through a temporary HTTPS tunnel and passed; that tunnel was not production hosting. No genuine Razorpay provider delivery appeared in dashboard logs.”

### 4:50–5:00 — Close

Restate: calibrated decision support, grounded evidence, safe Test Mode integration, and human control. Show the release verification result if time permits.

## Stop services

Press `Ctrl+C` in the backend and frontend terminals. No automatic refund, capture, dispute, evidence submission, or customer contact needs to be undone.

## Claims to avoid

- real production data, production effectiveness, guaranteed savings, or real-world fraud detection;
- a 30-day target—the target is 120 days;
- causal SHAP interpretation or treating explanation as evidence;
- saying a customer committed fraud or had malicious intent;
- saying the support-enhanced model improved every metric—precision decreased slightly;
- Razorpay Live Mode, real money, automatic capture/refund/dispute actions, or Test Mode results inside held-out metrics;
- genuine provider webhook delivery, a stable deployment, or production hosting unless independently verified after this runbook was written.
