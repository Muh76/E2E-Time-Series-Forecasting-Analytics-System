# End-to-End Time Series Forecasting & Analytics System

A production-oriented forecasting platform that takes raw store-level sales data through ETL, feature engineering, model training, and API serving. Forecasts are produced by deterministic models (LightGBM, Seasonal Naive); an LLM-based copilot explains results — it does not perform predictions.

---

## System Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Raw Data   │───▶│  ETL Pipeline    │───▶│  Feature Engine  │───▶│  Model Training  │
│  (CSV/DB)   │    │  (clean, augment)│    │  (lags, rolling, │    │  (LightGBM +     │
│             │    │                  │    │   calendar)       │    │   Seasonal Naive) │
└─────────────┘    └──────────────────┘    └──────────────────┘    └────────┬─────────┘
                                                                            │
                                                                            ▼
┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Streamlit  │◀───│  FastAPI Backend │◀───│  Model Loader    │◀───│  Artifact Store  │
│  Dashboard  │    │  (REST API)      │    │  (startup load)  │    │  (.joblib, .json)│
└─────────────┘    └────────┬─────────┘    └──────────────────┘    └──────────────────┘
                            │
                   ┌────────┴─────────┐
                   │  LLM Copilot     │
                   │  (explain only)  │
                   └──────────────────┘
```

All forecasts come from deterministic, auditable models. The LLM copilot provides natural-language explanations over precomputed results — it never generates predictions.

---

## Data Flow

```
1. Ingest       data/etl/ingest.py          Raw CSV → data/raw/
2. Clean        data/etl/clean.py           Nulls, types, outliers → cleaned DataFrame
3. Augment      data/etl/augment.py         Store metadata join, promo flags
4. Validate     data/etl/validate.py        Schema and range checks
5. ETL Output   scripts/run_etl.py          → data/processed/etl_output.parquet
6. Features     data/feature_engineering/   Lags [1,7,14], rolling [7,14], calendar
7. Train        scripts/train.py            Time-split, fit LightGBM + baseline
8. Artifacts    artifacts/models/           .joblib models, feature_columns.json, metadata
9. Serve        backend/app/main.py         FastAPI loads artifacts at startup
10. Predict     POST /api/v1/forecast/store  Feature pipeline → recursive predict → response
```

The same feature engineering pipeline (`data/feature_engineering/`) runs during both training and inference, ensuring train/serve parity. Feature columns are persisted as `feature_columns.json` and strictly enforced at prediction time.

---

## Recursive Forecasting

For multi-step horizons (horizon > 1), the system uses autoregressive recursive prediction:

1. **Step 1**: Use full observed history to compute features (lags, rolling stats). Predict one step ahead.
2. **Step h**: Append the step h-1 prediction to the history as if it were observed. Re-run the feature pipeline on a trailing window to recompute lags and rolling features from the updated series. Predict step h.
3. **Repeat** until the full horizon is covered.

This ensures that lag and rolling features reflect previous predictions rather than stale historical values, preventing the common multi-step forecasting bug where all steps return identical predictions.

Implementation: `LightGBMForecast._predict_one_series()` in `models/forecasting/lightgbm_forecast.py` maintains a `current_base` DataFrame with only raw columns. At each step it appends a placeholder row, re-runs `run_feature_pipeline()` on a small tail window, predicts, then writes the result back.

---

## Leakage Prevention

Data leakage is prevented at multiple layers:

| Layer | Guard |
|-------|-------|
| Feature selection | `_NEVER_FEATURE_COLS = frozenset({"target_raw"})` — columns that must never be used as features, regardless of presence in the DataFrame |
| Feature column builder | `_get_feature_columns()` automatically excludes date, target, entity, and all never-feature columns |
| Post-fit validation | `train.py` raises `ValueError` if `target_raw` appears in `_feature_cols` after fitting |
| Consistency check | `train.py` verifies `model.feature_name_` matches `_feature_cols` exactly before saving artifacts |
| Inference enforcement | `_enforce_feature_columns()` validates that served features match training features in name and order |

The `target_raw` column (raw sales before cleaning) was identified as a severe leakage source — its feature importance gain was orders of magnitude higher than any legitimate feature. It is now permanently blocked.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health/live` | Liveness probe |
| `GET` | `/api/v1/model/info` | Model metadata (version, features, training dates, metrics) |
| `POST` | `/api/v1/forecast/store` | Generate horizon-step forecast for a store |
| `POST` | `/api/v1/forecast/store/debug` | Forecast debug metadata (no inference) |
| `POST` | `/api/v1/backtest/store` | Rolling-origin backtesting for a store |

### Request Validation

All endpoints enforce strict Pydantic validation:

- **`store_id`**: Must exist in the processed dataset (checked against cached store ID set)
- **`horizon`**: 1 to 60 (inclusive)
- **`n_splits`**: 1 to 20 (backtest only)
- **History length**: Must exceed the model's `max_lag` before forecasting proceeds

Invalid requests return structured 422 errors with per-field messages.

---

## Example Requests & Responses

### Forecast

```bash
curl -X POST http://localhost:8000/api/v1/forecast/store \
  -H "Content-Type: application/json" \
  -d '{"store_id": 1, "horizon": 3}'
```

```json
{
  "store_id": 1,
  "horizon": 3,
  "forecasts": [
    {"date": "2015-08-01", "forecast": 5234.17, "confidence_low": 4102.83, "confidence_high": 6365.51},
    {"date": "2015-08-02", "forecast": 4891.42, "confidence_low": 3760.08, "confidence_high": 6022.76},
    {"date": "2015-08-03", "forecast": 5102.88, "confidence_low": 3971.54, "confidence_high": 6234.22}
  ]
}
```

Confidence intervals use `forecast +/- 1.96 * residual_std` (95% level) from training residuals.

### Model Info

```bash
curl http://localhost:8000/api/v1/model/info
```

```json
{
  "model_version": "v3",
  "trained_at": "2026-01-28T14:30:00Z",
  "feature_count": 22,
  "max_lag": 14,
  "lookback_window": 14,
  "train_start": "2013-01-01",
  "train_end": "2015-06-19",
  "train_rows": 811615,
  "train_rmse": 576.3421,
  "train_mae": 345.8912,
  "residual_std": 576.3421
}
```

### Backtest

```bash
curl -X POST http://localhost:8000/api/v1/backtest/store \
  -H "Content-Type: application/json" \
  -d '{"store_id": 1, "horizon": 7, "n_splits": 3}'
```

```json
{
  "store_id": 1,
  "n_splits": 3,
  "horizon": 7,
  "splits": [
    {"split": 1, "cutoff_date": "2014-08-15", "horizon": 7, "rmse": 1245.32, "mae": 982.45, "mape": 14.32},
    {"split": 2, "cutoff_date": "2014-11-20", "horizon": 7, "rmse": 1102.87, "mae": 876.19, "mape": 12.85},
    {"split": 3, "cutoff_date": "2015-02-25", "horizon": 7, "rmse": 1310.56, "mae": 1045.78, "mape": 15.10}
  ],
  "average": {"rmse": 1219.58, "mae": 968.14, "mape": 14.09}
}
```

### Validation Error (422)

```json
{
  "detail": [
    {
      "field": "store_id",
      "message": "Value error, store_id=9999 does not exist in the dataset. Valid range: 1–1115 (1115 stores).",
      "type": "value_error",
      "input": 9999
    }
  ]
}
```

---

## Model Governance & Metadata

Every training run produces `artifacts/models/model_metadata.json`:

| Field | Description |
|-------|-------------|
| `model_version` | Auto-incremented (`v1`, `v2`, ...) |
| `trained_at` | UTC ISO timestamp |
| `feature_count` | Number of features used |
| `feature_columns` | Ordered list of feature names |
| `max_lag` | Largest lag feature (e.g., 14) |
| `lookback_window` | Window size required for feature computation |
| `train_start` / `train_end` | Date range of training data |
| `train_rows` | Number of valid training samples |
| `train_rmse` / `train_mae` | In-sample fit metrics |
| `residual_std` | Standard deviation of training residuals (used for confidence intervals) |

Additional artifacts saved per run:

- `primary_lightgbm.joblib` — serialized LightGBM model
- `baseline_seasonal_naive.joblib` — serialized baseline model
- `feature_columns.json` — ordered feature list for inference enforcement
- `metrics.json` — validation set metrics for both models

### Reproducibility

Training is fully deterministic:

- `random.seed(42)` (Python stdlib)
- `numpy.random.seed(42)` (NumPy)
- `random_state=42` + `deterministic=True` (LightGBM)

---

## Backtesting

The system implements rolling-origin backtesting via `POST /api/v1/backtest/store`:

1. Load the store's full history and run the feature pipeline once.
2. Compute `n_splits` cutoff dates spaced evenly across the available date range. Each cutoff leaves at least `horizon` dates ahead for evaluation and sufficient history behind for feature computation.
3. For each split, pass history up to the cutoff into `model.predict()` (the pre-trained production model is used — no retraining per split).
4. Align forecasted dates with actual values, compute RMSE, MAE, and MAPE per split.
5. Return per-split metrics and averages.

This measures how the fixed production model generalizes across different time periods, rather than overfitting evaluation to a single holdout window.

---

## How to Reproduce Locally

### Prerequisites

- Python 3.11+
- Raw data in `data/raw/` (Rossmann store sales dataset)

### 1. Install dependencies

```bash
pip install pandas numpy lightgbm scikit-learn joblib pyyaml fastapi uvicorn requests
```

### 2. Run ETL pipeline

```bash
python scripts/run_etl.py
```

Produces `data/processed/etl_output.parquet`.

### 3. Train models

```bash
python scripts/train.py
```

Produces artifacts in `artifacts/models/` (model files, feature columns, metadata, metrics).

### 4. Start API server

```bash
uvicorn backend.app.main:app --reload
```

Models are loaded at startup. API is available at `http://localhost:8000`.

### 5. Run smoke tests

```bash
python scripts/smoke_test_api.py
```

Validates `/api/v1/model/info`, `/api/v1/forecast/store`, and `/api/v1/forecast/store/debug`.

### 6. Start frontend (optional)

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

### Configuration

All pipeline parameters are in `config/base/default.yaml`. Environment overrides (`local`, `staging`, `prod`) merge on top. Key sections: `feature_engineering` (lags, rolling windows), `training` (split, seed), `model` (horizon, frequency).

---

## Project Structure

```
├── config/                     Configuration (base + env overrides)
├── data/
│   ├── etl/                    Extractors, transformers, loaders
│   ├── feature_engineering/    Lag, rolling, calendar pipelines
│   ├── raw/                    Landing zone (gitignored)
│   └── processed/              ETL output (gitignored)
├── models/
│   ├── forecasting/            LightGBM, Seasonal Naive, base interface
│   ├── evaluation/             Metrics (MAE, RMSE, MAPE)
│   └── monitoring/             Drift and performance checks
├── backend/
│   └── app/
│       ├── main.py             FastAPI app, startup loader, error handler
│       ├── api/v1/             Routers: forecast, backtest, model_info, copilot
│       └── services/           Forecasting, backtest, model loading, RAG
├── frontend/                   Streamlit dashboard (pages, components)
├── copilot/                    LLM agents and prompts
├── scripts/                    train.py, run_etl.py, inference.py, smoke_test_api.py
├── artifacts/models/           Serialized models and metadata (gitignored)
├── tests/                      Unit, integration, e2e
├── docs/                       Architecture, API contract, data contract
└── infra/                      GCP Terraform, CI/CD workflows
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ETL & Features | Python, Pandas, config-driven pipelines |
| Forecasting | LightGBM (primary), Seasonal Naive (baseline) |
| Backend API | FastAPI, Pydantic v2, Uvicorn |
| Frontend | Streamlit, Plotly |
| LLM Copilot | OpenAI API (explanation only) |
| Serialization | Joblib (models), JSON (metadata) |
| Infrastructure | GCP (Cloud Run, GCS, Secret Manager), Terraform, GitHub Actions |

---

For detailed layer responsibilities, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). For API schema details, see [docs/API_CONTRACT.md](docs/API_CONTRACT.md).
