# Reproducibility and Release

## Supported environment

- Windows PowerShell 7+ for the documented local workflow
- Python 3.12+; final local verification also covers Python 3.14
- Node.js 22+ and npm using `frontend/package-lock.json`
- Docker for the portable backend image

Python direct dependencies are pinned in `requirements.txt`; frontend dependencies are locked by npm. The synthetic generator, split, LightGBM configuration, support processing, and ablation use seed 42 where randomness applies.

## Clean setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
npm --prefix frontend ci
Copy-Item .env.example .env
Copy-Item frontend/.env.example frontend/.env
```

Secret values in `.env.example` are blank. Release validation sets all optional AI and Razorpay network integrations off, so it requires no credential and performs no external financial call.

## Regeneration order

Generated data and the model/calibrator binaries are ignored. Rebuild the canonical baseline from a clean checkout:

```powershell
.\.venv\Scripts\python.exe -m src.pipeline
```

Regenerate Phase 7 governance artifacts in this order when their inputs or code change:

```powershell
.\.venv\Scripts\python.exe -m src.explain
.\.venv\Scripts\python.exe -m src.model_explanation_service
.\.venv\Scripts\python.exe -m src.support_events
.\.venv\Scripts\python.exe -m src.support_signal_report
.\.venv\Scripts\python.exe -m src.ablation
.\.venv\Scripts\python.exe -m src.model_card
```

`src.pipeline` records runtime durations and generation timestamps in the manifest, so those provenance fields may differ while the fixed-seed data, split, model, and metrics remain deterministic.

## One-command release gate

```powershell
.\scripts\verify_release.ps1
```

The gate fails immediately on a missing stage and checks:

- required model/report files and model version `0.1.0`;
- exact `chargeback_within_120d` target and isotonic calibration;
- 12,173-row held-out ID alignment across predictions and explanations;
- `shap-v1` additivity, factor shape, direction, ordering, and forbidden fields;
- `plain-v1` text plus the exact **Model explanation — not evidence** label;
- exactly three safe support signals, 8,270 observable events, and 5,297 future exclusions;
- canonical ablation baseline and model-card/report agreement;
- OpenAPI generation, unique operation IDs, and all eight Razorpay routes;
- ignored/untracked secret, SQLite, runtime, cache, and frontend build paths;
- credential-like strings and temporary public tunnel URLs in tracked text;
- the complete backend suite, frontend lint and production build, and `git diff --check`.

The expected canonical calibrated AP is 0.1492642172 and Brier score is 0.0062581254. Maximum SHAP reconstruction error must remain at or below `1e-8` (the current report is around machine precision). The ablation baseline must match the canonical report exactly.

Isotonic calibration can assign the same probability to many payments. Precision-at-fraction metrics break only those ties with an outcome-blind deterministic BLAKE2b hash of the unique payment ID; payment identity does not enter model fitting or probability estimation. This avoids platform-dependent ordering of equal NumPy values.

The only accepted test warnings are current third-party SHAP/Matplotlib notices about colormap methods and LightGBM TreeExplainer output shape. Project-owned deprecations, duplicate operation IDs, TypeScript errors, or failed assertions are release failures.

## CI

`.github/workflows/ci.yml` runs on pushes and pull requests with read-only repository permission. It installs Python 3.12 and Node 22, regenerates the ignored baseline artifacts, installs frontend dependencies with `npm ci`, and invokes the same PowerShell release gate. AI, Razorpay integration, and webhooks remain disabled. No real secret is defined or required.

## Production backend image

```powershell
docker build -t chargeback-radar-api .
docker run --rm -p 8000:8000 chargeback-radar-api
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

The image installs `libgomp1` for LightGBM, regenerates ignored model/data artifacts during build, runs as a non-root user, exposes port 8000, includes a health check, and starts Uvicorn without `--reload`.

For an external Docker host:

1. authenticate to the chosen provider and create a service from the root `Dockerfile`;
2. set `BACKEND_ALLOWED_ORIGINS` to the exact deployed frontend HTTPS origin;
3. keep all optional integrations disabled for the public synthetic demo, or add Test Mode secrets only through the provider secret manager;
4. verify `GET https://<backend-host>/api/health` returns `status=healthy`;
5. build the frontend with `VITE_API_URL=https://<backend-host>` and publish `frontend/dist` to a static host;
6. verify the dashboard loads from the stable frontend URL and inspect its generated assets for absence of server-secret names/values.

No hosting provider or paid plan is selected by the repository. A temporary public tunnel is not a deployment.
