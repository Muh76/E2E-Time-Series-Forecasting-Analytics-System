# 15-Day Agile Execution Plan

**Scope:** End-to-end time series forecasting & LLM insight copilot  
**Style:** Execution-focused; phases overlap where dependencies allow.

---

## Timeline Overview

| Days   | Focus                    |
|--------|--------------------------|
| 1–3    | Data engineering         |
| 2–5    | Modeling                 |
| 4–6    | Monitoring               |
| 6–8    | LLM integration          |
| 7–11   | Backend                  |
| 9–12   | Frontend                 |
| 12–15  | Deployment & integration |

---

## Phase 1: Data Engineering (Days 1–3)

**Goals**
- Ingest raw data into a controlled store (local/GCS).
- Run ETL: clean, align, normalize time series.
- Produce a feature pipeline (lags, rolling, seasonality) usable by train and serve.

**Key deliverables**
- Extractors for at least one source (file/DB or GCS).
- Transformers: validation, frequency alignment, dedup.
- Loaders writing to `data/processed` (and optional feature_store).
- Feature pipeline (config-driven) with train/serve parity.
- `scripts/run_etl.py` runnable end-to-end.
- Unit tests for core ETL and feature steps.

**Risks**
- Source schema or availability changes mid-sprint → lock schema contract and sample dataset.
- Feature drift between train and serve → single pipeline code path, config-only params.

---

## Phase 2: Modeling (Days 2–5)

**Goals**
- Train at least one deterministic model (e.g. Prophet or ARIMA) from processed/feature data.
- Run inference (batch or on-demand) and write forecasts to a defined output.
- Version model artifacts and config.

**Key deliverables**
- Model interface/abstract base in `models/forecasting/`.
- At least one concrete model implementation wired to config.
- `scripts/train.py`: load config, run training, save artifact + metadata.
- `scripts/inference.py`: load model and config, read features, write forecasts.
- Serialized model and run metadata under `artifacts/`.
- Unit tests for train/inference with small fixture data.

**Risks**
- Underfitting or bad default params → fix horizon and frequency in config; add a minimal backtest in evaluation phase.
- Slow training → cap data size or steps for Day 5; optimize later.

---

## Phase 3: Monitoring (Days 4–6)

**Goals**
- Compute performance metrics (e.g. MAE, MAPE) over holdout or recent data.
- Optional: simple drift indicator (e.g. distribution shift) and threshold check.
- Expose metrics for API and alerts (no alert delivery required in 15 days).

**Key deliverables**
- Performance calculator (metrics from forecast vs actual) and thresholds from config.
- Drift module stub or simple implementation (e.g. PSI or bin-based comparison).
- Monitoring config: thresholds, window, schedule (cron or trigger).
- Output: metrics and drift status consumable by backend (file or in-memory for Day 6).
- Tests for metric and threshold logic.

**Risks**
- Drift definition unclear → ship a single, simple metric; refine later.
- No production store yet → design interface (e.g. “get latest metrics”) so backend can swap storage later.

---

## Phase 4: LLM Integration (Days 6–8)

**Goals**
- Call one LLM provider (e.g. OpenAI) with API key from env.
- Build a small “copilot” service that receives context (forecast + metadata only) and returns an explanation (no prediction).
- Keep all prediction logic in the modeling layer.

**Key deliverables**
- Client wrapper: provider, model, temperature, max_tokens, timeout from config.
- Prompt templates and a single “explain” flow (e.g. “Explain this forecast change” with context).
- Copilot module: input = query + context payload; output = explanation + sources.
- No raw series sent to LLM; context limited to precomputed forecasts and summary stats.
- Unit test with mocked LLM response.

**Risks**
- Rate limits or latency → respect timeout and retries from config; optional caching later.
- Hallucination → prompt “answer only from provided context”; list sources in response.

---

## Phase 5: Backend (Days 7–11)

**Goals**
- FastAPI app with health, config loading, and the core API contract implemented.
- Endpoints: generate forecast, historical metrics, monitoring summary/series, copilot explain (and optional summarize).
- No direct DB in scope if not needed; use files or in-memory for 15-day scope.

**Key deliverables**
- App entrypoint, config loader (merge base + env), and structured errors.
- `GET /health/live`, `GET /health/ready`.
- `POST /api/v1/forecasts/generate` → call inference, return contract response.
- `GET /api/v1/metrics/historical` (and optional POST) → return from stored metrics.
- `GET /api/v1/monitoring/summary`, `GET /api/v1/monitoring/series`.
- `POST /api/v1/copilot/explain` (and optional `/copilot/summarize`) → call copilot, return explanation + sources.
- Request/response shapes match `docs/API_CONTRACT.md`.
- Integration tests for at least two endpoints with test data.

**Risks**
- Scope creep on auth/RBAC → defer; optional API key or IAM at deployment.
- Blocking on “real” storage → use file-backed or in-memory adapters with a clear interface to swap later.

---

## Phase 6: Frontend (Days 9–12)

**Goals**
- Streamlit app that talks only to the backend API.
- Show forecasts and historical metrics; optional monitoring view.
- Integrate copilot: user asks a question, app calls backend explain, displays answer.

**Key deliverables**
- `frontend/app.py` (or multi-page) with navigation.
- Pages: forecast view (series selector, horizon, “Generate” → call API, chart); historical metrics (filters, table/chart); optional monitoring summary.
- Copilot UI: input box, “Ask” → `POST /copilot/explain`, display explanation and sources.
- API base URL from config/env; no hardcoded secrets.
- Basic error handling and loading states.

**Risks**
- Backend contract changes → keep frontend to contract; version API if needed.
- Poor UX with slow LLM → show loading state; consider async or “Summarize” shortcut later.

---

## Phase 7: Deployment (Days 12–15)

**Goals**
- Run backend and frontend in containers (Dockerfiles).
- CI: lint, unit tests, optional integration tests on push/PR.
- Deploy to GCP (e.g. Cloud Run) for backend and frontend; config via env and Secret Manager for API keys.
- One successful end-to-end run: ETL → train → inference → API → frontend → copilot.

**Key deliverables**
- Dockerfile(s) for backend and frontend (or combined if simple).
- CI workflow (e.g. GitHub Actions): checkout, install, lint, test; optional build and push images.
- GCP: Terraform or manual setup for Cloud Run (or equivalent), GCS for data/artifacts if used, Secret Manager for `LLM_API_KEY`.
- Env-specific config (staging/prod) loaded via `APP_ENV` or equivalent.
- Runbook or short doc: how to run ETL, train, inference, and what URLs to hit.
- Day 15: smoke test full path; fix critical bugs; document known gaps.

**Risks**
- Permissions or quota on GCP → confirm project and roles early (Day 12).
- Flaky tests in CI → isolate and fix or skip with a ticket; don’t block deploy.

---

## Daily Focus (Suggested)

| Day | Primary focus |
|-----|----------------|
| 1   | ETL extractors + raw store |
| 2   | ETL transform/load + feature pipeline; model interface + train script |
| 3   | Feature pipeline tests; train one model; inference script |
| 4   | Monitoring performance + thresholds; drift stub |
| 5   | Train/inference with config; monitoring wiring |
| 6   | Monitoring tests; LLM client + prompts; copilot “explain” |
| 7   | FastAPI app + config; health + forecast generate |
| 8   | API: metrics, monitoring, copilot endpoints; copilot tests |
| 9   | Streamlit shell + forecast page; API client |
| 10  | Historical metrics + monitoring pages |
| 11  | Copilot UI; backend integration tests |
| 12  | Dockerfiles; CI workflow; GCP project/prep |
| 13  | Deploy backend + frontend to GCP; secrets + env |
| 14  | E2E test in deployed env; runbook; fix blockers |
| 15  | Smoke test; document gaps and next steps |

---

## Success Criteria for Day 15

- Raw data → ETL → features → train → inference → forecasts stored.
- API serves forecasts, historical metrics, monitoring summary, and copilot explain per contract.
- Frontend displays forecasts and copilot answers using only the API.
- Backend and frontend run on GCP with env-based config and no secrets in repo.
- CI runs lint and tests on push/PR.
