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

---

## 9. Rossmann Store Sales — Data Contract

This section defines the data contract for the Rossmann store sales pipeline: source files, join logic, normalized schema, cleaning rules, and output guarantees.

### 9.1 Source files

| File        | Description                    |
|-------------|--------------------------------|
| `train.csv` | Historical sales and store-day attributes (e.g. Open, Promo, StateHoliday). |
| `store.csv` | Store master data (e.g. StoreType, Assortment, CompetitionDistance).     |

### 9.2 Join logic

- **Join:** `train` **LEFT JOIN** `store` on `train.Store` = `store.Store`.
- All rows from `train` are kept; store attributes are attached where the store key matches. Rows with no matching store get NULLs in store columns (downstream cleaning may drop or fill).

### 9.3 Normalized internal schema (used by all downstream layers)

After join and cleaning, the single internal DataFrame used by ETL and downstream layers has the following columns:

| Column                 | Type      | Notes                                      |
|------------------------|-----------|--------------------------------------------|
| `date`                 | datetime  | Daily; observation date.                   |
| `store_id`             | int       | Store identifier (from `train.Store`).     |
| `target_raw`           | float     | Original Sales from source.                |
| `target_cleaned`       | float     | Sales after cleaning rules (see 9.4).     |
| `open`                 | bool      | Store open (1) or closed (0).              |
| `promo`                | bool      | Promotion running that day.                |
| `state_holiday`        | category  | State holiday indicator.                   |
| `school_holiday`       | bool      | School holiday indicator.                  |
| `store_type`           | category  | Store type (from store master).            |
| `assortment`           | category  | Assortment type (from store master).       |
| `competition_distance` | float     | Distance to competitor; missing filled (see 9.4). |

### 9.4 Cleaning rules

- **Rows with Open == 0:**  
  - **Choice:** `target_cleaned` is set to **0** (not dropped). Rationale: preserve the (store_id, date) grain for downstream reindex and feature alignment; closed days have no demand, so 0 is the correct cleaned target. Dropping would create date gaps and complicate daily reindex.
- **Sales == 0 and Open == 0:**  
  - Not treated as demand; they are closed days. `target_cleaned` = 0.
- **Missing CompetitionDistance:**  
  - Filled with the **median** of non-null CompetitionDistance (per store or global, as configured).
- **Categorical columns:**  
  - `state_holiday`, `store_type`, and `assortment` are cast explicitly to a categorical dtype (e.g. `pd.Categorical` or string with documented levels).

### 9.5 Output

- **Artifact:** A single DataFrame.
- **Sort order:** Sorted by `(store_id, date)`.
- **Uniqueness:** No duplicate `(store_id, date)` keys.
- **Persistence:** Saved as Parquet to `data/processed/etl_output.parquet`.
