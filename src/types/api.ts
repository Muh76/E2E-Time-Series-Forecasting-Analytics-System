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

export interface FeatureImportanceItem {
  feature: string;
  importance: number;
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
  feature_importance: FeatureImportanceItem[];
  max_lag: number;
  lookback_window: number;
}

// ---------------------------------------------------------------------------
// Monitoring
// ---------------------------------------------------------------------------

export interface MonitoringPerformance {
  mae: number | null;
  rmse: number | null;
  mape: number | null;
  sample_size: number;
  source?: string;
}

export interface MonitoringDriftIndicator {
  feature: string;
  score: number;
}

export interface MonitoringDrift {
  status: string;
  last_checked: string;
  indicators: MonitoringDriftIndicator[];
  overall_score: number | null;
  threshold: number;
  per_feature_scores: Record<string, number>;
}

export interface MonitoringPipeline {
  last_training: string | null;
  last_etl: string | null;
  status: string;
}

export interface MonitoringAlerts {
  mae: boolean;
  mape: boolean;
  drift: boolean;
}

export interface MonitoringThresholds {
  mae_alert: number;
  mape_alert: number;
  drift_threshold: number;
}

export interface MonitoringRollingSeries {
  mae: { date: string; value: number }[];
  mape: { date: string; value: number }[];
}

export interface MonitoringSummary {
  model_version: string;
  as_of: string;
  performance: MonitoringPerformance;
  drift: MonitoringDrift;
  pipeline: MonitoringPipeline;
  rolling_series: MonitoringRollingSeries;
  alerts: MonitoringAlerts;
  thresholds: MonitoringThresholds;
  overall_status: string;
}

// ---------------------------------------------------------------------------
// Copilot
// ---------------------------------------------------------------------------

export interface CopilotSource {
  type: string;
  note?: string;
}

export interface CopilotExplainRequest {
  query: string;
  context?: Record<string, unknown>;
  options?: Record<string, unknown>;
}

export interface CopilotExplainResponse {
  answer: string;
  reasoning: string;
  confidence: number;
  intents: string[];
  explanation: string;
  sources: CopilotSource[];
  generated_at: string;
  generator: string;
  openai_skipped: boolean;
  openai_skip_reason: string | null;
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

export interface ChatSource {
  source: string;
  header: string;
  score: number;
}

export interface ChatRequest {
  message: string;
}

export interface ChatResponse {
  status: string;
  message_received: string;
  reply: string;
  sources: ChatSource[];
  generated_at: string;
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
