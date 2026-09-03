Chargeback Radar converts calibrated payment risk and merchant-side operational failures into explainable, lowest-expected-cost defensive recommendations—without automatically taking a financial action.

Built for Razorpay AI Buildathon 2026 · Track 02: AI Risk Manager.

Why this exists

A conventional fraud score answers only one question: “How risky is this payment?” A merchant still needs to decide whether to monitor it, prepare evidence, review it, or recommend a refund—and every unnecessary intervention has a customer and operational cost.

Chargeback Radar joins three layers:

A calibrated LightGBM model estimates capture-time chargeback probability.

Deterministic rules identify explainable payment and merchant-operation failures.

A policy engine selects the defensive action with the lowest expected cost under declared assumptions.

The system is explicitly defense-only. It produces decision support, requires human approval for consequential actions, and never executes a refund automatically.

Five-minute judge path

Open the dashboard and inspect held-out precision, recall, calibration and false-positive cost.

Change review cost and intervention-effectiveness assumptions in the policy lab.

Compare conservative, expected and optimistic benefit scenarios.

Open a transaction to see its calibrated probability, deterministic rule findings and TreeSHAP factors.

Confirm that every metric comes from the untouched test set through reports/submission_manifest.json.

Architecture

flowchart TD
    A["Synthetic payments + delayed outcomes"] --> B["Customer-disjoint time split"]
    B --> C["Capture-time features"]
    C --> D["LightGBM risk model"]
    D --> E["Probability calibration"]
    E --> F["Expected-cost policy"]
    G["Payment + lifecycle events"] --> H["Deterministic rule engines"]
    H --> I["Explainable findings"]
    D --> J["TreeSHAP factors"]
    I --> K["Human approval boundary"]
    J --> K
    F --> K
    K --> L["Monitor · Evidence · Review · Refund recommendation"]

Honest held-out results

The following block is generated from reports/submission_manifest.json. Do not edit the values manually; run python -m src.readme_metrics or the full pipeline.

<!-- CHARGEBACK_RADAR_METRICS:START -->
| Measure | Generated result |
|---|---:|
| Synthetic payments | 80,000 |
| Synthetic chargebacks | 578 |
| Overall base rate | 0.722% |
| Held-out payments | 12,173 |
| Held-out chargebacks | 87 |
| Average Precision | 14.93% |
| Random/base-rate baseline | 0.715% |
| AP lift over baseline | 20.9× |
| Precision | 19.47% |
| Recall | 58.62% |
| F1 | 0.2923 |
| Brier score before calibration | 0.015723 |
| Brier score after calibration | 0.006258 |
| False-positive count | 211 |
| Detector false-positive cost | ₹31,650 |
| Policy gross avoided loss | ₹202,250 |
| Policy intervention cost | ₹65,382 |
| Policy estimated net benefit | ₹136,868 |
| Effectiveness sensitivity range | ₹92,805 to ₹177,416 |

_Generated from the fixed-seed pipeline; monetary values are synthetic held-out backtest estimates._
<!-- CHARGEBACK_RADAR_METRICS:END -->

Accuracy is intentionally not used as a headline metric because chargebacks are rare. Average Precision, precision, recall, calibration and false-positive cost are more informative for this imbalanced problem.

Evaluation design

Partition

Time window

Purpose

Train

Months 1–9

Fit preprocessing and LightGBM

Calibration

Month 10

Select and fit probability calibration

Held-out test

Months 11–12

Report final model and policy results

Customer IDs are disjoint across all three partitions. The test set is untouched during model fitting and calibration. A fixed seed makes the full pipeline reproducible.

Only information available at payment capture enters the ML model. Chargeback reason, dispute date/status, future delivery and future refund outcomes are forbidden model inputs. Post-payment shipment and refund events live in data/operations.parquet and are evaluated separately at an explicit as_of timestamp.

Decision policy

For each transaction and action, the policy estimates:

total expected cost
  = expected remaining chargeback loss
  + direct intervention cost
  + expected legitimate-customer friction

It then selects the least-cost defensive recommendation:

Action

Purpose

Automatic execution?

MONITOR

Observe low-risk payments

No action required

PREPARE_EVIDENCE

Assemble defensible payment evidence

No

MANUAL_REVIEW

Route uncertain cases to an operator

No

RECOMMEND_REFUND

Recommend refund for clear merchant errors

Never

Gross avoided loss, remaining chargeback loss, intervention cost, false-positive cost and net benefit are reported separately. LOW/BASE/HIGH effectiveness scenarios are assumptions—not confidence intervals or guaranteed savings.

Deterministic verification

Capture-time rules include duplicate charges, cancelled-subscription billing, authentication gaps, device/network conflicts, velocity spikes, repeat disputes, unclear descriptors and high-value new accounts.

Post-payment rules include:

SHIPMENT_SLA_BREACHED: activates only after a promised shipment deadline and recommends evidence preparation.

REFUND_NOT_PROCESSED: activates only after a promised refund deadline and can recommend a refund with human approval.

The lifecycle engine deliberately ignores precomputed outcome flags and reconstructs what was knowable at the supplied as_of time.

Reproduce everything

Backend setup

python -m venv .venv

Windows PowerShell:

.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

macOS/Linux:

source .venv/bin/activate
pip install -r requirements.txt

Regenerate every dataset, artifact and report:

python -m src.pipeline

This finishes by writing reports/submission_manifest.json, including artifact sizes and SHA-256 hashes.

Run the API:

uvicorn src.api:app --reload

API documentation is available at http://127.0.0.1:8000/docs.

Frontend setup

cd frontend
npm install
npm run dev

Open http://127.0.0.1:5173.

Verification

python -m pytest -q
cd frontend
npm run build

The Vite chunk-size notice is currently a non-blocking performance warning; it is recorded in notes/what_broke.md rather than hidden.

Important API routes

Route

Purpose

GET /api/health

Service health

GET /api/metrics

Model, calibration and default-policy results

POST /api/score

Score one payment and return defensive guidance

POST /api/simulate

Recalculate policy economics

GET /api/transactions

Review the policy-ranked queue

GET /api/policy/frontier

Review-cost sensitivity

GET /api/policy/effectiveness

LOW/BASE/HIGH effectiveness sensitivity

Repository map

src/
  generate.py                  synthetic customers and payments
  outcomes.py                  delayed disputes and reason codes
  operations.py                isolated shipment/refund lifecycle events
  features.py                  capture-time feature engineering
  train.py                     customer-disjoint LightGBM training
  calibrate.py                 probability calibration
  evaluate.py                  held-out metrics and curves
  decide.py                    expected-cost action policy
  rules.py                     capture-time deterministic rules
  post_payment_rules.py        time-aware lifecycle rules
  explain.py                   TreeSHAP explanations
  policy_lab.py                review-cost frontier
  effectiveness_sensitivity.py assumption stress test
  pipeline.py                  one-command reproducibility
frontend/                      React + TypeScript dashboard
reports/                       generated evaluation evidence
artifacts/                     generated model metadata and bundles
tests/                         leakage, policy, rules and pipeline tests
notes/what_broke.md            honest engineering incident log

Limitations

All current data and monetary outcomes are synthetic backtests.

No claim is made about realised savings on Razorpay production traffic.

Intervention effectiveness is declared and stress-tested, not learned from production experiments.

Synthetic behaviour can validate architecture and evaluation discipline, but not real-world generalisation.

The current workflow is decision support and keeps consequential actions behind human approval.

Build status

Implemented: reproducible synthetic data, leakage-safe evaluation, calibrated model, deterministic rules, TreeSHAP, cost policy, sensitivity analysis, API, dashboard and automated tests.

Planned submission hardening: AI-assisted evidence drafting, audit/fallback controls, Razorpay Test Mode adapter, deployment QA and the final five-minute demo.