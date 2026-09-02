# Chargeback Radar

Chargeback Radar is a defense-only chargeback risk detector built for Razorpay AI Buildathon 2026, Track 02: AI Risk Manager.

It predicts whether a captured card payment may become a chargeback, explains the important risk signals, and recommends the lowest-cost defensive action.

## Defensive actions

- Monitor
- Prepare evidence
- Manual review
- Recommend refund with human approval

## Core principles

- LightGBM performs risk prediction.
- Probabilities are calibrated before financial decisions.
- Evaluation uses a time-based held-out test set.
- Precision, recall, Average Precision and false-positive cost are reported.
- Sensitive actions require human approval.
- Synthetic data is clearly disclosed.
- No offensive fraud capability is included.

## Technology

- Python
- LightGBM
- scikit-learn
- SHAP
- FastAPI
- React
- TypeScript
- Recharts

## Development status

Phase 1: Project foundation completed.
