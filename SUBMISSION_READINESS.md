# Chargeback Radar Submission Readiness

Technical release status as of 2026-09-05. Phase 10 (recording and hackathon submission) is intentionally not included.

## Release decision

Phases 8 and 9 passed independent review and the complete local release gate. Chargeback Radar remains a defense-only prototype evaluated on synthetic data; the reported metrics do not establish production performance.

## Verified canonical results

| Item | Verified value |
|---|---:|
| Model version | `0.1.0` |
| Target | `chargeback_within_120d` |
| Held-out rows | 12,173 |
| Held-out positives | 87 |
| Base rate | 0.714696% |
| Calibrated Average Precision | 0.149264 |
| Precision | 19.466% |
| Recall | 58.621% |
| Brier score | 0.006258 |
| Confusion matrix (TN / FP / FN / TP) | 11,875 / 211 / 36 / 51 |
| False-positive review cost | INR 31,650 at INR 150 per review |
| Support-signal AP delta | +0.003508 |
| SHAP maximum reconstruction error | 0.0000000000 |

## Release validation

- Full reproduction: `python -m src.pipeline` — PASS, 17 ordered offline-safe stages in 107.9 seconds.
- Backend: `python -m pytest -q` — PASS, 514 passed and 5 accepted third-party SHAP/Matplotlib warnings in 11.74 seconds.
- Frontend lint: `npm --prefix frontend run lint` — PASS.
- Frontend production build: `npm --prefix frontend run build` — PASS; TypeScript and Vite completed, largest emitted chunk 369.92 kB.
- OpenAPI: PASS, 23 paths and 23 unique operation IDs. Seven secure Razorpay routes are mounted; direct payment-ID scoring is absent.
- Evidence red-team: PASS, 13 of 13 scenarios.
- Artifact determinism: PASS; repeat SHAP and evidence generations produced identical SHA-256 hashes.
- Repository hygiene: PASS; Git whitespace, tracked runtime-file, credential-pattern, and temporary-tunnel checks passed.
- Independent review: PASS after correcting unsigned-webhook status handling, legacy cached-score compatibility, and configurable Razorpay SQLite placement; no blocking findings remain.
- Docker: configuration statically reviewed; build not executed because the local Docker Desktop Linux daemon was unavailable.
- Deployment: not performed because the repository contains no selected/authenticated hosting provider. No stable URL is claimed.

## Razorpay Test Mode controls

- Test Mode and INR only; Live Mode configuration is rejected.
- Checkout HMAC is verified server-side with provider order/payment binding before scoring.
- The unsafe `/api/razorpay-test/payments/{payment_id}/score` route is unavailable.
- Webhooks verify the unmodified raw body, enforce event idempotency, and persist metadata only.
- Razorpay demo rows never enter canonical training, calibration, test, or ablation reports.
- No capture, refund, transfer, dispute submission, or customer message is executed.
- Human approval remains mandatory and `action_executed` remains false.

## Key artifacts

- `README.md`
- `MODEL_CARD.md`
- `reports/model_card.json`
- `reports/submission_manifest.json`
- `reports/metrics.json`
- `reports/calibration_metrics.json`
- `reports/ablation.json`
- `reports/evidence_guardrail_eval.json`
- `docs/DEMO_RUNBOOK.md`

The release commit is identified by `git rev-parse HEAD` after the approved changes are committed. The manifest records the exact source revision used for its reproducibility run without creating a self-referential commit hash.

## Remaining Phase 10 work

Select fixed demo cases, record and upload the 1080p video, confirm public repository and deployment links, run final incognito checks, complete the hackathon form, and save submission confirmation.
