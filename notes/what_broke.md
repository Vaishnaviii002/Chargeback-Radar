What broke while building Chargeback Radar

This is an engineering log, not a marketing document. It records failures encountered during the hackathon build, their causes, the fixes applied and the checks added to prevent recurrence.

1. The API root returned 404 Not Found

Symptom: Uvicorn started normally, but opening http://127.0.0.1:8000/ returned 404.

Cause: Only API routes had been defined; no root route existed.

Fix: Added a small root response pointing to /docs and retained /api/health for machine-readable health checks.

Lesson: A healthy server and a defined root route are different things. Smoke-test the exact URLs used in the demo.

2. pytest could not import src

Symptom: Test collection failed with ModuleNotFoundError: No module named 'src' even though commands such as python -m src.features worked.

Cause: The repository root was not on pytest's import path in that Windows environment.

Fix: Added pytest.ini with the project root as pythonpath and ran tests from the repository root.

Prevention: The documented test command is always python -m pytest -q from chargeback-radar.

3. The first leakage guard rejected threeds_status

Symptom: Feature generation stopped with Future-data leakage detected in feature: threeds_status.

Cause: The original guard used an over-broad text match for the word status. threeds_status is known at capture time and is legitimate; dispute status is not.

Fix: Replaced the substring heuristic with an explicit forbidden-field list and capture-time feature allow-list.

Prevention: Tests verify that chargeback reason, dispute timestamps/status and future fulfilment/refund outcomes cannot enter MODEL_FEATURES.

4. The initial split allowed customer overlap

Symptom: Time splitting alone allowed customers seen during training to reappear in calibration or test data.

Risk: Reported performance could partially measure customer memorisation rather than generalisation.

Fix: Assigned customers to disjoint train, calibration and held-out groups while preserving the month 1–9 / month 10 / months 11–12 evaluation windows.

Prevention: Split assertions and model metadata now verify zero customer-ID overlap.

5. A manually transferred Git patch failed

Symptom: git apply first reported a corrupt patch and later reported hunks that did not apply.

Cause: Chat/browser-to-editor transfer altered patch formatting, while some target files had also changed locally.

Fix: Checked the worktree, applied clean hunks carefully and moved subsequent changes as complete validated files rather than fragile large patches.

Lesson: Validate with git apply --check first, preserve local edits, and prefer small changes or complete files during time-constrained remote pairing.

6. Frontend build ran from the wrong directory

Symptom: npm run build returned ENOENT for chargeback-radar/package.json.

Cause: The Vite project lives under frontend, not the repository root.

Fix: Changed into frontend before running npm commands.

Prevention: Backend and frontend commands are separated explicitly in the README.

7. TypeScript policy output did not match the API

Symptom: PolicySimulator.tsx failed with Property 'costs' does not exist on type 'SimulationResult'.

Cause: The backend had moved monetary results under a costs object, while the frontend type still represented an earlier response shape.

Fix: Updated the TypeScript response contract and all consumers to use the nested cost fields.

Prevention: Production builds run TypeScript compilation before Vite bundling.

8. An API edit introduced indentation errors

Symptom: Pylance reported unexpected indentation, an incomplete try statement and missing blocks around the scoring route.

Cause: A partial manual paste landed inside an existing block at the wrong indentation level.

Fix: Reconstructed the entire affected function, checked for duplicate dictionary keys and ran python -m py_compile src/api.py before starting Uvicorn.

Lesson: For Python control-flow edits, compile the file before relying on editor diagnostics or runtime reload.

9. A corrupted TSX paste produced hundreds of errors

Symptom: PolicyFrontier.tsx produced more than 400 TypeScript errors beginning on one line.

Cause: Non-code prose was accidentally pasted into a TypeScript expression, causing a parser cascade.

Fix: Replaced the component with a clean validated file and ran both TypeScript and Vite production builds.

Lesson: Hundreds of errors on one line usually indicate one parser break, not hundreds of independent defects.

10. The first shipment generator was unrealistic

Symptom: It generated 10,675 shipment SLA breaches across 80,000 payments—roughly one fifth of shippable orders.

Cause: Normal delivery variance plus late and never-delivered probabilities compounded into an implausibly unhealthy merchant population.

Fix: Tightened normal delivery timing and reduced explicit late/never-delivered rates. The regenerated dataset produced 3,739 shipment breaches, 2,763 refund requests and 371 refund SLA breaches.

Prevention: Operational summaries are reviewed as domain distributions, not merely checked for non-zero values.

11. Model and policy metrics were initially easy to confuse

Symptom: The detector showed one flagged count and precision/recall pair, while the cost policy showed another.

Cause: The detector uses a fixed operating threshold; the policy independently chooses among four actions by expected cost.

Fix: Labelled them separately in the UI and documented that detector classification metrics and policy-intervention metrics answer different questions.

12. Monetary benefit needed an auditable decomposition

Symptom: The policy originally displayed baseline cost, policy cost and net benefit but did not show how much loss was avoided versus spent on interventions.

Fix: Added remaining chargeback loss, gross avoided loss and intervention cost. The identity is now testable:

policy cost = remaining chargeback loss + intervention cost
net benefit = gross avoided loss - intervention cost

LOW/BASE/HIGH effectiveness scenarios were also added so the displayed result is not tied to one optimistic assumption.

13. Vite reports a chunk-size warning

Symptom: The frontend production build succeeds but warns that the main JavaScript chunk exceeds 500 kB after minification.

Cause: Recharts and the dashboard currently ship in one bundle.

Decision: Accepted temporarily because it does not affect correctness or the local demo. Route/component-level lazy loading remains a deployment-hardening task.

Status: Known, non-blocking and intentionally disclosed.