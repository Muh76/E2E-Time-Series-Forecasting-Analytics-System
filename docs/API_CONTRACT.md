# API Contract: Time Series Forecasting & Analytics Backend

**Document type:** API specification (contract only; no implementation)  
**Base path (example):** `/api/v1`  
**Content type:** `application/json` unless noted.

---

## 1. Health Checks

### 1.1 Liveness

**Purpose:** Indicate that the service process is running. Used by orchestrators (e.g. Kubernetes, Cloud Run) to decide whether to restart the container.

| Item | Value |
|------|--------|
| **Method** | `GET` |
| **Path** | `/health/live` |
| **Request** | None (no body, no required query params) |
| **Response** | See below |

**Response payload:**

```json
{
  "status": "ok"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Always `"ok"` when the process is alive |

**Status codes:** `200` only when live; `503` if the process is shutting down or unhealthy in a way that warrants restart.

---

### 1.2 Readiness

**Purpose:** Indicate that the service can accept traffic (e.g. DB/feature-store reachable, model loaded). Used by load balancers or orchestrators to stop sending traffic during startup or when dependencies are unavailable.

| Item | Value |
|------|--------|
| **Method** | `GET` |
| **Path** | `/health/ready` |
| **Request** | None |
| **Response** | See below |

**Response payload:**

```json
{
  "status": "ready",
  "checks": {
    "feature_store": "ok",
    "model_registry": "ok"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"ready"` when all checks pass; `"not_ready"` otherwise |
| `checks` | object | Optional. Map of dependency name to status (`"ok"` or error detail) |

**Status codes:** `200` when ready; `503` when not ready.

---

## 2. Generating Forecasts

### 2.1 Generate forecast (on-demand)

**Purpose:** Request a new forecast for one or more time series, for a given horizon and (optionally) model version. The backend runs inference and returns point forecasts and optional intervals.

| Item | Value |
|------|--------|
| **Method** | `POST` |
| **Path** | `/api/v1/forecasts/generate` |
| **Request** | JSON body below |
| **Response** | JSON body below |

**Request payload:**

```json
{
  "series_ids": ["series_001", "series_002"],
  "horizon_steps": 12,
  "frequency": "D",
  "model_version": "v2.1.0",
  "options": {
    "return_interval": true,
    "interval_level": 0.95
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `series_ids` | array of string | yes | Identifiers of the time series to forecast |
| `horizon_steps` | integer | yes | Number of steps to forecast (e.g. 12 for 12 days if frequency is daily) |
| `frequency` | string | no | Series frequency (e.g. `"D"`, `"H"`, `"W"`). Default from config. |
| `model_version` | string | no | Model version id; if omitted, default/active version is used |
| `options` | object | no | See below |

**options**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `return_interval` | boolean | false | If true, response includes prediction intervals |
| `interval_level` | number | 0.95 | Confidence level for intervals (e.g. 0.95) when `return_interval` is true |

**Response payload:**

```json
{
  "job_id": "gen_a1b2c3d4",
  "status": "completed",
  "forecasts": [
    {
      "series_id": "series_001",
      "model_version": "v2.1.0",
      "frequency": "D",
      "point_forecast": [100.5, 102.1, 99.8],
      "lower": [95.2, 96.8, 94.1],
      "upper": [105.8, 107.4, 105.5],
      "steps": [1, 2, 3]
    }
  ],
  "generated_at": "2025-01-30T14:00:00Z"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | string | Id of this generation run (for idempotency or async lookup) |
| `status` | string | `"completed"` for sync; `"accepted"` if request is queued for async processing |
| `forecasts` | array | One object per requested series (see below) |
| `generated_at` | string (ISO 8601) | Server time when forecasts were produced |

**forecasts[]**

| Field | Type | Description |
|-------|------|-------------|
| `series_id` | string | Same as in request |
| `model_version` | string | Version actually used |
| `frequency` | string | Frequency used |
| `point_forecast` | array of number | Point forecast per step |
| `lower` | array of number | Optional; lower bound of interval when requested |
| `upper` | array of number | Optional; upper bound of interval when requested |
| `steps` | array of integer | Step indices (1-based or 0-based; contract should fix one) |

**Status codes:** `200` success; `400` invalid request (e.g. unknown series_id, invalid horizon); `404` model version not found; `422` validation error; `503` if dependency (e.g. model) unavailable.

---

## 3. Retrieving Historical Metrics

**Purpose:** Return stored historical metrics (e.g. actuals, past forecasts, errors) for one or more series and a time range. Used for backtesting, dashboards, and comparison.

### 3.1 Get historical metrics

| Item | Value |
|------|--------|
| **Method** | `GET` or `POST` |
| **Path** | `/api/v1/metrics/historical` |
| **Request** | Query params (GET) or JSON body (POST) for complex filters |
| **Response** | JSON body below |

**Request (GET query parameters):**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `series_ids` | string | yes | Comma-separated list of series identifiers |
| `start_date` | string | yes | Start of range (ISO 8601 date or datetime) |
| `end_date` | string | yes | End of range (ISO 8601 date or datetime) |
| `metrics` | string | no | Comma-separated: e.g. `actual,forecast,error`. Default: all available |
| `model_version` | string | no | Filter by model version; if omitted, latest or all depending on policy |

**Request (POST body, alternative):**

```json
{
  "series_ids": ["series_001", "series_002"],
  "start_date": "2025-01-01",
  "end_date": "2025-01-28",
  "metrics": ["actual", "forecast", "error"],
  "model_version": "v2.1.0"
}
```

**Response payload:**

```json
{
  "data": [
    {
      "series_id": "series_001",
      "date": "2025-01-15",
      "actual": 98.2,
      "forecast": 100.1,
      "error": -1.9,
      "model_version": "v2.1.0"
    }
  ],
  "meta": {
    "series_ids": ["series_001"],
    "start_date": "2025-01-01",
    "end_date": "2025-01-28",
    "count": 28
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `data` | array | One row per (series_id, date); fields present only when requested and available |
| `meta` | object | Summary: series_ids, range, row count |

**Status codes:** `200` success; `400` invalid range or series; `422` validation error.

---

## 4. Retrieving Monitoring Data

**Purpose:** Expose model and pipeline health for operators: performance metrics, drift indicators, and pipeline status.

### 4.1 Get monitoring summary

| Item | Value |
|------|--------|
| **Method** | `GET` |
| **Path** | `/api/v1/monitoring/summary` |
| **Request** | Query params below |
| **Response** | JSON body below |

**Request (query parameters):**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_version` | string | no | Filter by model version; default: active or all |
| `since` | string | no | ISO 8601 datetime; only metrics after this time |

**Response payload:**

```json
{
  "model_version": "v2.1.0",
  "as_of": "2025-01-30T14:00:00Z",
  "performance": {
    "mae": 2.34,
    "rmse": 3.01,
    "mape": 0.042,
    "sample_size": 1200
  },
  "drift": {
    "status": "ok",
    "last_checked": "2025-01-30T13:00:00Z",
    "indicators": []
  },
  "pipeline": {
    "last_training": "2025-01-28T02:00:00Z",
    "last_etl": "2025-01-30T06:00:00Z",
    "status": "ok"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `model_version` | string | Version this summary applies to |
| `as_of` | string (ISO 8601) | Time of this summary |
| `performance` | object | Aggregate metrics over the monitored window |
| `drift` | object | Drift status, last check time, optional list of triggered indicators |
| `pipeline` | object | Last training/ETL times and overall pipeline status |

**Status codes:** `200` success; `404` if no data for the requested model/version.

---

### 4.2 Get monitoring time series (metrics over time)

**Purpose:** Return performance or drift metrics as a time series (e.g. daily MAE, weekly drift score) for charts and alerts.

| Item | Value |
|------|--------|
| **Method** | `GET` |
| **Path** | `/api/v1/monitoring/series` |
| **Request** | Query params |
| **Response** | JSON body below |

**Request (query parameters):**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `metric` | string | yes | One of: `mae`, `rmse`, `mape`, `drift_score`, etc. |
| `model_version` | string | no | Filter by model version |
| `start_date` | string | yes | Start of range (ISO 8601) |
| `end_date` | string | yes | End of range (ISO 8601) |
| `granularity` | string | no | `daily`, `weekly`; default `daily` |

**Response payload:**

```json
{
  "metric": "mae",
  "model_version": "v2.1.0",
  "granularity": "daily",
  "data": [
    { "date": "2025-01-28", "value": 2.1 },
    { "date": "2025-01-29", "value": 2.4 },
    { "date": "2025-01-30", "value": 2.3 }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `metric` | string | Echo of requested metric |
| `model_version` | string | Version used |
| `granularity` | string | Aggregation level |
| `data` | array | { date, value } pairs ordered by date |

**Status codes:** `200` success; `400` unknown metric or invalid range; `422` validation error.

---

## 5. Generating LLM-Based Explanations

**Purpose:** Ask the copilot to explain or summarize forecasts and metrics in natural language. The copilot uses only precomputed forecasts and metadata provided by the API; it does not perform prediction.

### 5.1 Generate explanation

| Item | Value |
|------|--------|
| **Method** | `POST` |
| **Path** | `/api/v1/copilot/explain` |
| **Request** | JSON body below |
| **Response** | JSON body below |

**Request payload:**

```json
{
  "query": "Why did the forecast for series_001 increase in the last run?",
  "context": {
    "series_ids": ["series_001"],
    "model_version": "v2.1.0",
    "forecast_job_id": "gen_a1b2c3d4",
    "include_metrics": true
  },
  "options": {
    "max_tokens": 512,
    "format": "plain"
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | yes | Natural-language question or instruction |
| `context` | object | no | Scope for the copilot (see below) |
| `options` | object | no | See below |

**context**

| Field | Type | Description |
|-------|------|-------------|
| `series_ids` | array of string | Limit explanation to these series |
| `model_version` | string | Forecasts/metrics from this version |
| `forecast_job_id` | string | Tie explanation to a specific generation job |
| `include_metrics` | boolean | If true, include recent performance metrics in context sent to LLM |

**options**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_tokens` | integer | 512 | Upper bound on response length |
| `format` | string | `"plain"` | `"plain"` or `"markdown"` for response formatting |

**Response payload:**

```json
{
  "explanation": "The forecast for series_001 increased in the last run because ...",
  "sources": [
    { "type": "forecast", "job_id": "gen_a1b2c3d4", "series_id": "series_001" },
    { "type": "metrics", "model_version": "v2.1.0" }
  ],
  "generated_at": "2025-01-30T14:01:00Z"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `explanation` | string | Copilot’s answer (plain or markdown per request) |
| `sources` | array | References to forecasts/jobs/metrics used as context (for audit) |
| `generated_at` | string (ISO 8601) | Time the explanation was generated |

**Status codes:** `200` success; `400` invalid query or context; `422` validation error; `503` if copilot/LLM is unavailable.

---

### 5.2 Summarize (predefined summary type)

**Purpose:** Request a standard summary (e.g. “weekly forecast summary”, “risks and anomalies”) without a free-form query. Same contract principle: explanation only, grounded in API-provided data.

| Item | Value |
|------|--------|
| **Method** | `POST` |
| **Path** | `/api/v1/copilot/summarize` |
| **Request** | JSON body below |
| **Response** | JSON body below |

**Request payload:**

```json
{
  "summary_type": "weekly_forecast",
  "context": {
    "series_ids": ["series_001", "series_002"],
    "model_version": "v2.1.0",
    "as_of_date": "2025-01-30"
  },
  "options": {
    "max_tokens": 1024,
    "format": "markdown"
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `summary_type` | string | yes | One of: `weekly_forecast`, `risks_and_anomalies`, `executive_summary`, etc. |
| `context` | object | no | Same shape as in `/copilot/explain`; constrains which data is summarized |
| `options` | object | no | Same as in `/copilot/explain` |

**Response payload:**

```json
{
  "summary": "# Weekly forecast summary\n\n...",
  "summary_type": "weekly_forecast",
  "sources": [
    { "type": "forecast", "model_version": "v2.1.0" },
    { "type": "metrics", "model_version": "v2.1.0" }
  ],
  "generated_at": "2025-01-30T14:02:00Z"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `summary` | string | Generated summary (plain or markdown per options) |
| `summary_type` | string | Echo of requested type |
| `sources` | array | Data used for the summary |
| `generated_at` | string (ISO 8601) | Generation time |

**Status codes:** `200` success; `400` unknown summary_type or invalid context; `422` validation error; `503` copilot unavailable.

---

## 6. Summary Table

| Purpose | Method | Path |
|---------|--------|------|
| Liveness | GET | `/health/live` |
| Readiness | GET | `/health/ready` |
| Generate forecast | POST | `/api/v1/forecasts/generate` |
| Historical metrics | GET / POST | `/api/v1/metrics/historical` |
| Monitoring summary | GET | `/api/v1/monitoring/summary` |
| Monitoring time series | GET | `/api/v1/monitoring/series` |
| LLM explanation | POST | `/api/v1/copilot/explain` |
| LLM summarize | POST | `/api/v1/copilot/summarize` |

---

## 7. Conventions

- **Versioning:** API version in path (`/api/v1/...`). New backward-incompatible behavior introduced under a new version path.
- **Errors:** Problem details (e.g. RFC 7807 or a simple `{ "detail": "...", "code": "..." }`) for 4xx/5xx with consistent structure.
- **Ids:** Use opaque strings (e.g. `job_id`, `series_id`); avoid overloading with business meaning.
- **Dates/times:** ISO 8601 (e.g. `2025-01-30`, `2025-01-30T14:00:00Z`). Document whether times are UTC.
- **Pagination:** For list-style endpoints (e.g. historical metrics with large result sets), add `limit`/`offset` or `cursor` in a later iteration of this contract.
