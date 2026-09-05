# Security and Safety

Chargeback Radar supports merchant defense; it is not an autonomous fraud adjudicator. Its recommendations never prove wrongdoing, customer intent, causation, or the factual basis of a dispute.

## Secrets and configuration

- `.env`, `frontend/.env`, runtime stores, and build output are ignored.
- `RAZORPAY_KEY_SECRET`, `RAZORPAY_WEBHOOK_SECRET`, `OPENAI_API_KEY`, and `EVIDENCE_AUDIT_HMAC_KEY` stay server-side and must be supplied through a local ignored file or deployment secret manager.
- The frontend may receive only `VITE_API_URL`; the public Razorpay Test key is returned for a server-created order. No server secret belongs in a `VITE_` variable.
- Razorpay integration and webhooks are opt-in. `RAZORPAY_TEST_MODE=false` and non-`rzp_test_` keys fail closed. Dependency configuration failures return sanitized 503 responses.
- `BACKEND_ALLOWED_ORIGINS` accepts comma-separated HTTP(S) origins only. Wildcards, paths, fragments, queries, and credential-bearing URLs are rejected while credentialed CORS is enabled.

## Model and data controls

Customer identity is used only to enforce disjoint evaluation and never enters the model. Outcome labels, dispute state/reason, future chargebacks, future delivery, and future refunds are forbidden model inputs. Merchant context for a Test Mode payment must be timestamped no later than the provider payment time.

SHAP records bind to the authoritative model, preprocessor, version, feature order, and held-out IDs. Contributions describe model behavior only. Every delivered explanation includes **Model explanation — not evidence** and denies causal, fraud, and intent conclusions.

## AI boundary

All AI integrations default off. Evidence generation sends only approved trusted facts; exact citations, payment ID, recommended action, draft-only messages, human approval, and `action_executed=false` are enforced after generation. Support extraction removes PII, separates instructions from data, limits input, excludes identity/outcomes/future events, and caches by a sanitized input hash. Raw support text is not stored in the signal report. Provider failure selects a deterministic fallback.

## Checkout and provider binding

Checkout signatures use HMAC-SHA256 over `server_order_id|razorpay_payment_id` with the Test secret and `hmac.compare_digest`. Verification happens before provider fetch or scoring. Server lookups then confirm both IDs, order link, amount, INR, card method, and authorized/captured status. The adapter exposes create/fetch operations only; it implements no capture, refund, dispute, or customer-contact method.

## Webhooks and idempotency

Webhook HMAC-SHA256 covers the exact raw bytes and uses constant-time comparison before JSON parsing. The endpoint bounds buffered input to 1 MB. Only supported event types are processed; unsupported signed events are recorded as ignored. Atomic SQLite reservations make identical delivery a replay and changed content under the same event ID a conflict.

Only minimal metadata is stored. Even verified events execute no payment, refund, dispute, evidence-submission, or customer action. Human approval remains required.

## Known limitations

- The dataset and cost results are synthetic and do not demonstrate real-world security or model effectiveness.
- A genuine Razorpay-originated provider webhook was not observed; signed local and temporary-tunnel requests were verified.
- A temporary tunnel is not production hosting and its URL is not committed.
- SQLite is suitable for this demo and single-instance deployment. A multi-instance production design would require a shared transactional replay store and operational retention policy.
- Optional third-party AI and Razorpay availability remain external dependencies when explicitly enabled.

Report suspected secret exposure by revoking the affected credential first, removing it from the deployment environment, and rotating any dependent audit/webhook configuration. Do not place a replacement secret in Git or chat.
