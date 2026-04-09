/**
 * Shared TypeScript types matching the backend API contract.
 *
 * Endpoints covered:
 *   GET  /api/v1/model/info
 *   POST /api/v1/forecast/store
 *   POST /api/v1/forecast/store/debug
 *   POST /api/v1/backtest/store
 */

// ---------------------------------------------------------------------------
// Forecast
// ---------------------------------------------------------------------------

export interface ForecastPoint {
  date: string;
  forecast: number;
  confidence_low: number | null;
  confidence_high: number | null;
}

export interface ForecastRequest {
  store_id: number;
  horizon: number;
}

export interface ForecastResponse {
  store_id: number;
  horizon: number;
  forecasts: ForecastPoint[];
}

// ---------------------------------------------------------------------------
// Forecast Debug
// ---------------------------------------------------------------------------

export interface ForecastDebugResponse {
  store_id: number;
  last_observed_date: string;
  model_version: string;
  feature_columns_used: string[];
  max_lag_used: number;
  lookback_window: number;
  recursive_steps: number;
}

// ---------------------------------------------------------------------------
// Backtest
// ---------------------------------------------------------------------------

export interface BacktestRequest {
  store_id: number;
  horizon: number;
  n_splits: number;
}

export interface BacktestSplit {
  split: number;
  cutoff_date: string;
  horizon: number;
  rmse: number;
  mae: number;
  mape: number;
}

export interface BacktestAverageMetrics {
  rmse: number;
  mae: number;
  mape: number;
}

export interface BacktestResponse {
  store_id: number;
  n_splits: number;
  horizon: number;
  splits: BacktestSplit[];
  average: BacktestAverageMetrics;
}

// ---------------------------------------------------------------------------
// Model Metadata
// ---------------------------------------------------------------------------

export interface TrainingDateRange {
  start: string;
  end: string;
}

export interface ValidationMetrics {
  rmse: number | null;
  mae: number | null;
  mape: number | null;
}

export interface ModelMetadata {
  model_version: string;
  trained_at: string;
  training_date_range: TrainingDateRange;
  feature_columns: string[];
  feature_count: number;
  sample_size: number;
  hyperparameters: Record<string, unknown>;
  residual_std: number;
  validation_metrics: ValidationMetrics;
  max_lag: number;
  lookback_window: number;
}

// ---------------------------------------------------------------------------
// Validation Error (structured 422)
// ---------------------------------------------------------------------------

export interface ValidationErrorItem {
  field: string;
  message: string;
  type: string;
  input: unknown;
}

export interface ValidationErrorResponse {
  detail: ValidationErrorItem[];
}
