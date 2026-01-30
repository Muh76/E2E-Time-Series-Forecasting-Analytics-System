# Configuration

Configuration is YAML-based with environment separation.

## Structure

- **`base/default.yaml`** — Full schema and placeholder values. All keys and defaults live here.
- **`local/config.yaml`** — Local development overrides (debug on, lower workers, local API URL).
- **`staging/config.yaml`** — Staging overrides (stricter thresholds, staging API URL, optional GCS URIs).
- **`prod/config.yaml`** — Production overrides (tighter thresholds, alert channels, prod API URL).

## Load order

1. Load `base/default.yaml`.
2. Load `{env}/config.yaml` (e.g. `local`, `staging`, `prod`) and deep-merge over base.
3. Override with environment variables where supported (e.g. `LLM_API_KEY` — never put secrets in YAML).

## Environment selection

Set `APP_ENV=local|staging|prod` (or equivalent) so the app loads the correct env file. Default can be `local` when unset.

## Schema overview

| Section       | Purpose |
|---------------|---------|
| `data`        | Paths for raw, processed, feature store, artifacts; optional cloud URIs |
| `model`       | Default model type, version tag, horizon, frequency, seasonality, interval |
| `training`    | Train/validation dates, split, optimization, checkpoint dir, early stopping |
| `evaluation`  | Metrics list, thresholds (mae_max, rmse_max, mape_max), holdout, promotion rules |
| `monitoring`  | Performance thresholds, drift threshold, window, schedule, alert cooldown and channels |
| `llm`         | Provider, api_key_env, model, temperature, max_tokens, timeout, retries |
| `api`         | env, debug, log_level, host, port, workers |
| `frontend`    | api_base_url, page_title |

Secrets (e.g. LLM API key) are referenced by env var name in config (`api_key_env`) and supplied at runtime.
