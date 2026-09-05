# Architecture

Chargeback Radar is a defense-only decision-support application with four deliberately separated data paths: canonical synthetic evaluation, model explanation, grounded evidence drafting, and Razorpay Test Mode operations.

## End-to-end flow

1. `src.generate`, `src.outcomes`, and `src.operations` create synthetic payments, delayed outcomes, and post-payment operations with seed 42.
2. `src.features` builds capture-time features and enforces the `chargeback_within_120d` target boundary.
3. `src.train` creates customer-disjoint months 1–9 / month 10 / months 11–12 partitions, fits preprocessing and LightGBM, and writes model version `0.1.0`.
4. `src.calibrate` selects isotonic calibration on month 10 and applies it to the untouched test set.
5. `src.evaluate`, `src.decide`, `src.policy_lab`, and `src.effectiveness_sensitivity` report rare-event metrics, calibrated policy results, review-cost sensitivity, and assumption scenarios.
6. `src.explain` applies TreeSHAP to the fitted preprocessing/model pipeline and verifies reconstruction for every held-out payment. `src.model_explanation_service` produces deterministic analyst text labeled **Model explanation — not evidence**.
7. `src.support_events`, `src.support_text`, and `src.support_signal_service` compile only timestamp-observable support events, redact PII, isolate prompt instructions, and emit exactly three binary signals. `src.ablation` compares the canonical model against a controlled enhanced model.
8. Evidence Copilot builds trusted facts from synthetic operational records, generates a cited draft through guarded optional AI or deterministic fallback, and appends tamper-evident audit metadata.
9. FastAPI exposes reports and analyst workflows. React/Vite renders the dashboard; it never receives server secrets, raw support text, model binaries, or webhook bodies.

## Razorpay Test Mode path

Order creation validates INR amounts and canonical idempotency input before the server-side adapter calls Razorpay. The browser receives only the public `rzp_test_` key and trusted order fields.

Checkout success is not trusted by itself. The backend verifies HMAC-SHA256 over the server-owned order ID and returned payment ID in constant time before provider lookup. It then binds provider order/payment IDs, amount, INR currency, card method, and authorized/captured status. Merchant-declared context is marked separately, must be timezone-aware, and cannot be observed after the payment timestamp. Only approved capture-time features enter the existing model and isotonic calibrator.

The primary frontend flow calls `verify-and-score`. Read-only order and payment lookup endpoints remain available for Test Mode diagnostics, but direct payment-ID scoring is not mounted: scoring requires the server-owned order ID, Checkout HMAC, and provider order/payment binding. All responses state that held-out evaluation is unaffected and no financial action was executed.

## Webhook path

The webhook endpoint streams exact raw request bytes into a 1 MB bounded buffer. HMAC verification occurs before JSON decoding. The parser applies an eight-event allowlist and extracts only safe resource IDs. SQLite atomically reserves each event ID against the exact payload hash, returns an idempotent replay for identical duplicates, and rejects changed content.

The `webhook_events` table contains only `event_id`, `payload_hash`, `event_type`, `status`, `received_at`, and `processed_at`. It contains no raw body, PII, customer text, card data, or secret. `RAZORPAY_DATABASE_PATH` selects a writable persistent-volume location for a single-instance deployment. Legacy cached scores that predate calibration-version metadata remain replayable with an explicit `legacy-unrecorded` provenance label; their probabilities and decisions are not altered.

## Trust boundaries

| Boundary | Trusted source | Enforced controls |
|---|---|---|
| Model input | Generated capture-time schema or verified Test payment plus merchant context | Exact feature order, forbidden-field checks, timezone checks |
| Explanation | Fitted model, preprocessor, held-out IDs | Version binding, SHAP additivity, direction/order checks, non-causal disclaimer |
| Evidence | Deterministic trusted facts | Exact citations, guardrails, audit chain, human approval |
| Browser/API | Public requests | Pydantic contracts, safe errors, exact CORS allowlist |
| Razorpay | Server-authenticated Test API and signed callbacks | Test-key enforcement, HMAC, provider binding, no mutation methods |
| Webhook storage | Verified allowlisted metadata | Raw-byte hash, atomic replay guard, metadata-only schema |

## Runtime artifacts

Canonical reports are tracked. Generated data, model/calibrator binaries, caches, audit logs, SQLite files, and frontend build output are ignored. CI and the backend Docker image regenerate the complete canonical artifact chain using `python -m src.pipeline`; local release verification validates it before starting tests.
