# End-to-End Time Series Forecasting & LLM Insight Copilot System

A production-oriented platform for time series forecasting and decision support. Forecasts are produced by deterministic models; an LLM-based copilot explains results and answers questions—it does not perform predictions.

---

## Problem Statement

Organizations need reliable forecasts (demand, capacity, revenue) and clear explanations for stakeholders. Common gaps:

- **Forecasts** are produced in silos (spreadsheets, one-off scripts) with no shared pipeline, versioning, or monitoring.
- **Explanations** are manual: analysts write narratives from the same numbers that executives see, with no consistent link between model output and language.
- **Decision-making** lacks a single place to view forecasts, compare scenarios, and get grounded answers to “why did this change?” or “what should we watch?”

This system addresses those gaps with a single pipeline from raw data to forecasts, model monitoring, an API and dashboard for consumption, and an LLM copilot that explains and interprets—without replacing the forecasting logic.

---

## High-Level Architecture

```
Sources (DB, GCS, APIs)
        │
        ▼
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│ Data ingestion    │────▶│ ETL & feature eng │────▶│ Forecasting       │
│ (raw store)       │     │ (processed/feat.) │     │ (deterministic)    │
└───────────────────┘     └───────────────────┘     └─────────┬─────────┘
                                                              │
        ┌─────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────┐     ┌───────────────────┐
│ Eval & monitoring │     │ Backend API       │◀──── LLM copilot (explanation only)
│ (metrics, alerts) │     │ (FastAPI)         │
└───────────────────┘     └─────────┬─────────┘
                                    │
                                    ▼
                          ┌───────────────────┐
                          │ Frontend dashboard│
                          │ (Streamlit)       │
                          └───────────────────┘
```

Data flows one way: ingestion → ETL/features → forecasting → API. The copilot and dashboard consume only the API (forecasts and metadata). Deployment and automation run on GCP with CI/CD.

---

## Core Features

- **Data ingestion:** Connectors for databases, object storage, and APIs; raw data landed in a controlled store.
- **ETL & feature engineering:** Cleaned, aligned time series; reusable feature pipelines (lags, rolling stats, seasonality, calendar); versioned for train and serve.
- **Forecasting:** Scripted training and inference; deterministic models only (e.g. statistical or ML); outputs stored and versioned.
- **Model evaluation & monitoring:** Backtesting, holdout metrics, drift and performance monitoring, alerts.
- **LLM insight copilot:** Natural-language Q&A over forecasts and metadata (e.g. “Why did the forecast change?”, “Summarize risks.”). **Explanation and interpretation only—no prediction.**
- **Backend API:** Single entry point for forecasts, metadata, and copilot; versioned, stateless, suitable for horizontal scaling.
- **Frontend dashboard:** Time series and forecast visualizations, filters, and access to the copilot.
- **CI/CD & GCP deployment:** Automated build, test, and deploy; infra-as-code; config and secrets managed per environment.

---

## Tech Stack

| Area | Choices |
|------|--------|
| Data / ETL | Python, config-driven pipelines; GCS / BigQuery as stores |
| Features | Versioned pipelines; feature store layout for train/serve parity |
| Forecasting | Deterministic models (e.g. stats/ML); serialized artifacts; scripted train/inference |
| Monitoring | Metrics and drift components; pluggable backends |
| Copilot | LLM integration over API context (forecasts + metadata only) |
| Backend | FastAPI |
| Frontend | Streamlit |
| Infra / deployment | GCP (Cloud Run, GCS, Secret Manager, etc.); Terraform; GitHub Actions (or equivalent) for CI/CD |

---

## Project Structure Overview

```
├── config/                 # Environment-specific config (base, dev, prod)
├── data/
│   ├── etl/                # Extractors, transformers, loaders
│   ├── feature_engineering/  # Feature pipelines and transformers
│   ├── raw/                # Landing zone for raw data (gitignored contents)
│   ├── processed/          # ETL output (gitignored contents)
│   └── feature_store/      # Feature store layout (gitignored contents)
├── models/
│   ├── forecasting/        # Model definitions and base interfaces
│   └── monitoring/         # Drift and performance monitoring
├── backend/                # FastAPI app (API, core, services)
├── frontend/               # Streamlit app (pages, components)
├── copilot/                # LLM agents and prompts (explanation only)
├── scripts/                # Entrypoints: train, inference, run_etl
├── infra/
│   ├── gcp/                # Terraform, Cloud Run configs
│   └── ci_cd/              # Workflows (e.g. GitHub Actions)
├── tests/                  # Unit, integration, e2e; fixtures
├── docs/                   # Architecture and design (e.g. ARCHITECTURE.md)
├── artifacts/              # Model artifacts (gitignored contents)
└── notebooks/              # Exploratory work only; not part of production path
```

Data, modeling, backend, frontend, and infra are separated. Production behavior is driven by scripts and config; notebooks are for exploration only.

---

## How the System Supports Decision-Making

- **Single source of truth:** One pipeline from raw data to forecasts, with versioned config and code, so stakeholders see consistent numbers.
- **Transparency:** Monitoring and evaluation show how forecasts perform over time; the copilot explains what changed and what to watch, grounded in actual model output.
- **Separation of roles:** Numbers come from deterministic, auditable models; the LLM only explains and interprets. No “black box” prediction from the copilot.
- **Scalability and reliability:** API and jobs are designed for cloud deployment (e.g. GCP); config and secrets are environment-aware so dev and prod stay consistent and secure.

---

## LLMs: Explanation Only, Not Prediction

**All forecasts are produced by deterministic models (e.g. statistical or ML).** The LLM copilot is used only for:

- Explaining why a forecast changed or what drove an error
- Summarizing trends, risks, and talking points
- Answering natural-language questions over precomputed forecasts and metadata

The copilot **does not** generate forecasts, replace the model layer, or ingest raw series to “predict.” Predictions are reproducible, traceable, and owned by the forecasting pipeline; the LLM augments communication and interpretation.

---

## Future Extensions

- **Additional data sources:** More connectors (e.g. Snowflake, Kafka) and streaming ingestion.
- **Model registry:** Versioned model and run metadata; promotion and rollback workflows.
- **Scenario comparison:** Multiple forecast runs (e.g. assumptions, horizons) and comparison in the dashboard and API.
- **Copilot enhancements:** RAG over internal docs and run logs; structured outputs (e.g. bullet summaries) for reports.
- **Multi-tenant / RBAC:** Tenants and role-based access in the API and dashboard.
- **Cost and latency:** Caching, batch inference schedules, and optional edge deployment for low-latency reads.

---

For detailed layer responsibilities and data flow, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
