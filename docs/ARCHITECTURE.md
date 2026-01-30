# System Architecture: End-to-End Time Series Forecasting & Decision-Support Platform

**Document type:** Internal design  
**Scope:** Data ingestion → forecasting → insight → API → dashboard → GCP deployment

---

## 1. Overview

The platform is a cloud-ready analytics system that ingests time series data, runs ETL and feature engineering, trains and serves forecasting models, monitors model health, and exposes a backend API and Streamlit dashboard. An LLM-based insight copilot explains results and supports interpretation only; it does not perform predictions.

---

## 2. Layer Responsibilities

### 2.1 Data Ingestion Layer

**Responsibility:** Ingest raw data from source systems into a controlled landing zone.

- **Inputs:** Databases (BigQuery, Cloud SQL), object storage (GCS), streaming (Pub/Sub), APIs, file drops.
- **Outputs:** Raw data in `data/raw` (or GCS buckets) with metadata (source, timestamp, schema).
- **Behavior:** Connectors/extractors pull or receive data on schedule or event; optional validation and schema checks; no business logic.

### 2.2 ETL & Feature Engineering

**Responsibility:** Clean, align, and transform raw series into analysis-ready datasets and features.

- **ETL:** Extract → validate → normalize (e.g. frequency, timezone) → deduplicate → load to `data/processed` or feature store.
- **Feature engineering:** Lags, rolling stats, seasonality indicators, calendar features, exogenous variables; pipelines are versioned and reusable for train and serve.
- **Outputs:** Processed series and feature tables consumed by the forecasting layer and (via API) by the copilot for context.

### 2.3 Forecasting Model Layer

**Responsibility:** Produce point and interval forecasts for business planning.

- **Inputs:** Processed time series and feature tables from ETL/feature engineering.
- **Behavior:** Training (offline, scripted) produces serialized models and configs; inference (batch or on-demand) runs deterministic logic only (e.g. ARIMA, Prophet, or ML models). No LLM in the prediction path.
- **Outputs:** Forecasts (and optional backtest metrics) stored in artifacts/DB and exposed via API.

### 2.4 Model Evaluation & Monitoring

**Responsibility:** Ensure forecasts remain accurate and detect degradation.

- **Evaluation:** Backtesting, holdout metrics (e.g. MAE, RMSE, MAPE), and optional A/B comparisons before promoting a model.
- **Monitoring:** Track performance over time (accuracy, bias), data drift (feature/distribution shifts), and pipeline health; alert on thresholds.
- **Outputs:** Metrics dashboards, alerts, and signals used to trigger retraining or rollback.

### 2.5 LLM Insight Copilot (Explanation Only)

**Responsibility:** Explain forecasts and support decision-making through natural language; **no forecasting or prediction**.

- **Inputs:** Precomputed forecasts, metadata, and (optionally) summary stats/context from the API—never raw series for “prediction” by the LLM.
- **Behavior:** Answers questions like “Why did the forecast change?”, “What drove last month’s error?”, “Summarize trends and risks.” Uses prompts and retrieval over stored results so responses are grounded in actual model output.
- **Outputs:** Explanations, summaries, and suggested talking points for stakeholders.

**Why the LLM is isolated from prediction logic:**

- **Determinism & audit:** Forecasts must be reproducible and auditable; LLMs are non-deterministic and not suitable as the system of record for numbers.
- **Regulatory & trust:** Decision-support often requires traceability (which model, which data, which version); keeping predictions in a dedicated model layer keeps a clear boundary.
- **Cost & latency:** Inference is cheap and fast; LLM calls are slower and more expensive—reserved for explanation and Q&A only.
- **Correctness:** Time series forecasting uses domain-specific math and validated code; the LLM’s role is to interpret and communicate, not to replace that logic.

### 2.6 Backend API Layer (FastAPI)

**Responsibility:** Single entry point for data, forecasts, and copilot.

- **Endpoints:** Configurable by env (e.g. dev/prod); health, auth, and API versioning (e.g. `/api/v1/`).
- **Capabilities:** Serve forecasts (by series, horizon, model version), expose metadata and metrics, proxy or orchestrate copilot requests with the right context (forecast + metadata only).
- **Behavior:** Stateless, horizontally scalable; talks to feature store, model registry, and copilot service; no direct access to raw data beyond what’s needed for context.

### 2.7 Frontend Dashboard (Streamlit)

**Responsibility:** Visualize forecasts and insights for business users.

- **Features:** Time series charts, forecast vs actual, model comparison, key metrics, and an embedded or linked copilot UI for Q&A.
- **Data source:** All data via the backend API (no direct DB or model access); supports filters (date range, series, model version).
- **Behavior:** Read-only; reflects what the API returns and supports export/reporting for decision-making.

### 2.8 CI/CD & Deployment (GCP)

**Responsibility:** Automate build, test, and deployment of code and (where applicable) model artifacts.

- **CI:** On commit/PR—lint, unit tests, integration tests; optional container build and push to Artifact Registry.
- **CD:** Deploy backend (e.g. Cloud Run), frontend (Cloud Run or static + Cloud Storage), scheduled jobs for ETL and training (Cloud Scheduler + Cloud Run/Composer), and infra (Terraform).
- **Secrets & config:** Secret Manager and env-specific config (e.g. `config/dev`, `config/prod`); no secrets in code.

---

## 3. Data Flow (High Level)

```
Sources → [Data ingestion] → raw store
                ↓
         [ETL / feature eng] → processed / feature store
                ↓
         [Forecasting model] → forecasts + metadata
                ↓
         [Eval & monitoring] → metrics, alerts
                ↓
         [Backend API] ←───────────────────────────────
                ↑                    ↑
                |                    |
    [Frontend dashboard]    [LLM copilot]
    (charts, filters)      (explanation only; reads forecasts + context from API)
```

- **Ingestion → ETL:** Raw data is the single source of truth; ETL reads from it and writes processed/feature data.
- **ETL → Model:** Training and inference consume only processed/feature data; no raw access in the model layer.
- **Model → API:** Forecasts and metadata are stored and served by the API; the copilot and dashboard both consume the API.
- **Copilot:** Receives only what the API provides (forecasts, metadata, optional summaries); never receives raw series to “predict”; it only explains and answers questions.

---

## 4. How This Supports Business Decision-Making

- **Single source of truth:** One pipeline from raw data to forecasts, with clear ownership (ingestion → ETL → model) and reproducibility (versioned config and code).
- **Transparency:** Monitoring and evaluation give confidence in numbers; the copilot makes those numbers interpretable (“why,” “what if,” “what to watch”).
- **Separation of roles:** Forecasting stays deterministic and auditable; the LLM augments understanding and communication without affecting the numbers.
- **Scalability and reliability:** API and frontend scale independently; ETL and training run as scheduled or event-driven jobs; GCP services (Cloud Run, Secret Manager, etc.) support production SLAs and security.
- **Iteration:** CI/CD and config management allow safe, repeatable changes to data, features, and models while keeping the architecture stable.

---

## 5. Summary Table

| Layer              | Responsibility                          | Consumes           | Produces                    |
|--------------------|----------------------------------------|--------------------|-----------------------------|
| Data ingestion     | Ingest raw data                        | Source systems     | Raw store                   |
| ETL & features     | Clean, transform, feature pipelines   | Raw store          | Processed / feature store   |
| Forecasting        | Train and run deterministic forecasts | Features           | Forecasts, metadata         |
| Eval & monitoring  | Backtest, monitor, alert               | Forecasts, data    | Metrics, alerts             |
| LLM copilot        | Explain and support Q&A only           | API (forecasts)    | Explanations, summaries     |
| Backend API        | Serve data and orchestrate copilot     | Store, models, API | HTTP responses              |
| Frontend           | Visualize and expose copilot           | API                | Dashboards, exports         |
| CI/CD (GCP)        | Build, test, deploy, infra             | Repo, config       | Running services, jobs      |
