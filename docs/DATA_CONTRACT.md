# Internal Data Contract: Time-Series Forecasting System

**Audience:** Engineers and downstream consumers of ETL output.  
**Status:** Internal; not a public API.

---

## 1. Scope

This contract describes the shape, semantics, and guarantees of time-series data produced and consumed by the ETL pipeline and forecasting stack. It applies to data flowing from raw ingestion through cleaning and optional augmentation. It does not cover feature-store schemas, model-artifact formats, or API request/response bodies.

---

## 2. Entities and Grain

- **Grain:** One row = one observation for one entity at one point in time.
- **Entity:** A single time series (e.g. one store, one product, one region). Identified by an entity key column (e.g. `store_id`). Single-series datasets may have no entity column; then one row = one observation at one time.
- **Time:** One column holds the observation time (date or datetime). No duplicate (time, entity) pairs in validated/cleaned output.

---

## 3. Required Columns (with types)

| Column           | Type              | Required | Notes |
|------------------|-------------------|----------|--------|
| Time column      | `datetime64[ns]`  | Yes      | Name configurable (e.g. `date`). Normalized to daily when using daily pipeline. |
| Target column    | numeric           | Yes      | Name: `target` (raw). One value per (time, entity). |
| Entity column    | any hashable      | Conditional | Required for multi-entity (e.g. retail) pipelines; name configurable (e.g. `store_id`). |

After cleaning, a second target column may exist: `target_cleaned`. After augmentation, columns `target_original` (copy of `target`) and `target_augmented` may exist. See sections 5 and 8.

---

## 4. Time Assumptions

- **Frequency:** Daily pipeline assumes or produces daily frequency. Gaps may be filled during cleaning (reindex + fill).
- **Monotonicity:** After validation, for each entity the time column is strictly increasing (no duplicate dates per entity; no backwards dates).
- **Timezone:** Contract does not fix timezone. Data may be timezone-naive or timezone-aware; downstream must not assume one without config. Normalization (e.g. to UTC or a single zone) is environment-specific.

---

## 5. Target Semantics

- **`target`:** Raw observed value. Not modified by cleaning or augmentation. May contain NaNs where rows were added (e.g. filled dates after reindex). Preserved in all pipeline modes.
- **`target_cleaned`:** Output of cleaning only. Filled missing dates per configured strategy (e.g. forward-fill or zero). Optional clipping applied to this column only. Same grain as `target`; may have more rows if reindex added dates.
- **`target_augmented`:** Output of augmentation only. Same grain as input. Synthetic changes (missing blocks, noise, trend) applied here. Input `target` (and optional `target_original`) unchanged. Present only when augmentation has been run.

---

## 6. Guarantees After ETL

Downstream may rely on:

- Required columns present with the types above.
- No duplicate (time, entity) keys in validated/cleaned output.
- Time column strictly monotonic per entity.
- `target` non-negative (when retail/validation rules are enabled).
- `target` and (if present) `target_cleaned` / `target_augmented` numeric; NaNs only where defined by contract (e.g. filled-in dates in `target`, or augmentation missing-block).
- Sort order: deterministic by (time, entity) when sort is applied by pipeline.
- Augmentation: rows touched by synthetic changes are marked in `augmentation_type` (see section 8).

---

## 7. Non-Guarantees

ETL does **not** guarantee:

- A specific timezone or that all data uses the same zone.
- No missing (time, entity) cells except when reindex/fill is explicitly run and documented.
- Bounds on `target` or `target_cleaned` beyond non-negativity when that rule is enabled.
- That `target_cleaned` or `target_augmented` exist (they are optional outputs).
- Idempotency of runs or stability of row order beyond (time, entity).
- Backward compatibility of column names or semantics across versions.

---

## 8. Augmentation Semantics

- **Purpose:** Synthetic changes for robustness or sensitivity analysis only. Not used as the system-of-record series for production forecasts.
- **Columns added:** `target_original` (copy of `target`), `target_augmented` (modified series), `augmentation_type` (per-row label).
- **Marking:** `augmentation_type` is a string per row. Values include `original` (unchanged), `missing_block`, `noise_shift`, `trend`, or comma-separated combinations when multiple apply. Downstream can filter or weight by these labels.
- **Determinism:** With fixed config and seed, augmentation is reproducible. Without seed, it is not.
- **Scope:** Only `target_augmented` is modified; `target` and `target_original` are never overwritten by augmentation.
