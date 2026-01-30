# ETL Pipeline

**Purpose:** Ingest raw time-series data, validate schema and quality, clean (reindex, fill, optional clip), and optionally augment for robustness. Output is ready for feature engineering and forecasting; no model logic lives here.

**Module responsibilities:** `ingest` — load CSV (generic or retail); `validate` — required columns, types, no duplicates, non-negative target, monotonic dates per entity; `clean` — normalize dates, reindex to daily, fill missing values, optional outlier clip; `augment` — optional synthetic changes (missing blocks, noise shift, trend) with deterministic seed; `pipeline` — orchestrates the above in order.

**Execution order:** Ingest → Validate → Clean → (optional) Augment. Config drives each step; pipeline returns a single DataFrame (no file writing in the pipeline module).

**Guarantees after ETL:** Required columns present; no duplicate (date, entity); time strictly monotonic per entity; target non-negative (when validation enabled); raw column `target` preserved; `target_cleaned` and (if augment run) `target_augmented` and `augmentation_type` added. See [docs/DATA_CONTRACT.md](../../docs/DATA_CONTRACT.md) for full contract.
