# Chargeback Radar Model Card

**Model-card version:** `model-card-v1`

**Model version:** `0.1.0`

**Target:** `chargeback_within_120d`

**Status:** Hackathon prototype evaluated on synthetic data

## Summary

Chargeback Radar is a defense-only merchant decision-support system. Its LightGBM model estimates the probability of the synthetic target `chargeback_within_120d`. The score supports human prioritization; it does not establish fraud, causation, wrongdoing or dispute evidence.

Every protected merchant action requires human approval. The system never automatically refunds, messages a customer or submits a dispute.

## Target definition

`chargeback_within_120d` means whether a payment receives a chargeback within the synthetic 120-day observation window.

This is a 120-day target. It must not be described as a 30-day prediction.

## Data and evaluation split

- Data source: synthetic payments, outcomes, operations and support events
- Total payments: 80,000
- Training: months 1–9 (59,048 rows)
- Calibration: month 10 (8,779 rows)
- Held-out test: months 11–12 (12,173 rows)
- Customers disjoint across all splits: **Yes**
- Held-out test customers unseen during training: **Yes**

The held-out test is also the unseen-customer stress test because every test customer is disjoint from training and calibration.

## Model and calibration

- Estimator: `LGBMClassifier`
- Seed: `42`
- Best iteration: `1`
- Calibration method selected from calibration-month data: `isotonic`
- Review capacity used to choose the operating threshold: 2.000%

Calibration-method selection uses the first 70% of month 10 for fitting candidates and the final 30% for choosing between sigmoid and isotonic calibration. Test labels are not used for calibration or threshold selection.

## Canonical held-out performance

| Metric | Value |
|---|---:|
| Test payments | 12,173 |
| Chargebacks | 87 |
| Base rate | 0.715% |
| Average Precision | 0.149264 |
| Precision | 19.466% |
| Recall | 58.621% |
| F1 | 0.292264 |
| Brier score | 0.006258 |
| Expected calibration error | 0.001502 |
| Precision at top 1% | 18.033% |
| Precision at top 5% | 11.166% |
| Operating threshold | 0.160131 |
| Flagged payments | 262 |
| Flagged rate | 2.152% |

Accuracy is intentionally not used as the headline metric because the held-out base rate is below one percent.

## Unseen-customer stress test

- Test customers: 2,543
- Overlap with training customers: 0
- Overlap with calibration customers: 0
- Test rows: 12,173
- Average Precision: 0.149264
- Recall: 58.621%

## Controlled support-signal ablation

Both variants use identical payment rows, train/calibration/test partitions, seed, LightGBM settings, early stopping, calibration selection and evaluation.

The enhanced model adds exactly:

- `intent_to_cancel`
- `non_receipt_complaint`
- `dissatisfaction`

| Metric | Baseline | Enhanced | Enhanced − baseline |
|---|---:|---:|---:|
| Average Precision | 0.149264 | 0.152772 | +0.003508 |
| Precision | 19.466% | 19.366% | -0.099% |
| Recall | 58.621% | 63.218% | +4.598% |
| F1 | 0.292264 | 0.296496 | +0.004232 |
| Brier score | 0.006258 | 0.006200 | -0.000058 |
| Calibration error | 0.001502 | 0.001241 | -0.000261 |
| Precision at top 1% | 18.033% | 22.951% | +4.918% |
| Precision at top 5% | 11.166% | 11.494% | +0.328% |

The enhanced result is reported without cherry-picking: Average Precision, recall and Brier score improved, while precision changed by -0.099%.

## Support-text safety boundary

- Scoring point: payment timestamp plus seven days
- Payments with observable support text: 8,270
- Future support events excluded: 5,297
- Unique sanitized model inputs: 37
- Input-hash caching enabled: **Yes**
- PII removal before the model boundary: **Yes**
- Prompt-injection filtering before the model boundary: **Yes**
- Final chargeback labels and dispute outcomes exposed to the text model: **No**

The LLM converts sanitized, timestamp-safe support text into three bounded booleans. It does not produce the chargeback-risk probability.

## Explainability

SHAP explains how trained model inputs contributed to the model estimate.

**Model explanation — not evidence**

SHAP does not prove causation, fraud, customer intent or the factual basis of a dispute. Model explanations remain visually and contractually separate from Evidence Copilot facts.

## Model features

- `card_network`
- `product_category`
- `cvv_result`
- `threeds_status`
- `amount_paise`
- `hour_of_day`
- `day_of_week`
- `is_weekend`
- `is_digital_good`
- `descriptor_clarity_score`
- `phone_verified`
- `email_verified`
- `account_age_days`
- `has_prior_order`
- `total_prior_orders`
- `prior_disputes_count`
- `days_since_last_order`
- `txns_last_1h`
- `txns_last_24h`
- `txns_last_7d`
- `amount_last_24h_paise`
- `device_is_new`
- `ip_country_matches_billing`
- `ip_is_proxy_or_vpn`
- `threeds_liability_shift`
- `billing_shipping_distance_km`

## Forbidden features

- `customer_id`
- `customer_name`
- `customer_email`
- `customer_phone`
- `dispute_id`
- `dispute_status`
- `dispute_reason`
- `reason_code`
- `chargeback_family`
- `chargeback_outcome`
- `true_fraud`
- `final_outcome`
- `was_won`
- `was_lost`
- `future_chargeback`
- `future_refund`
- `future_delivery`
- `future_dispute`
- `final_reason_code`

Customer identifiers may be used only to enforce customer-disjoint evaluation and are never model inputs.

## Intended use

- Prioritize merchant payments for human risk review.
- Prioritize preparation of factual chargeback evidence.
- Support bounded, cost-aware merchant defense workflows.
- Evaluate risk-system methodology on synthetic held-out data.

## Prohibited use

- Automatic accusation of a customer.
- Criminal or fraud determination.
- Creditworthiness or lending decisions.
- Identity profiling.
- Automatic customer punishment.
- Automatic refund execution.
- Automatic customer messaging.
- Automatic dispute submission.
- Offensive fraud or payment abuse.

## Human-control requirements

- All evidence is a draft.
- Human approval is required.
- No payment action is automatically executed.
- No customer message is automatically sent.
- No dispute is automatically submitted.
- Recommended actions remain bounded to `MONITOR`, `PREPARE_EVIDENCE`, `MANUAL_REVIEW` and `RECOMMEND_REFUND`.

## Known limitations

- The dataset, chargebacks, operational events and support messages are synthetic.
- ('Absolute performance does not establish real-world generalization to merchant traffic.',)
- Issuer, network and merchant behavior is simplified compared with production.
- Concept drift and live population drift have not been measured.
- Support-text signals are evaluated at a seven-day post-payment scoring point.
- Support-message templates simplify real language diversity and ambiguity.
- SHAP describes model contribution, not causation, fraud or dispute evidence.
- The LLM does not generate calibrated chargeback-risk probability.
- Razorpay test mode cannot fabricate a real issuer-generated chargeback.
- All protected merchant actions require human review and approval.

## Reproducibility sources

- `artifacts/model_bundle.joblib`
- `artifacts/model_metadata.json`
- `artifacts/feature_schema.json`
- `reports/metrics.json`
- `reports/calibration_metrics.json`
- `reports/test_explanations.parquet`
- `reports/support_signals.parquet`
- `reports/ablation.json`
- `reports/model_card.json`
