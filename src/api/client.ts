import axios, { AxiosError, type AxiosInstance } from "axios";
import type {
  BacktestRequest,
  BacktestResponse,
  ForecastDebugResponse,
  ForecastRequest,
  ForecastResponse,
  ModelMetadata,
  ValidationErrorItem,
} from "../types/api";

// ---------------------------------------------------------------------------
// Typed API error
// ---------------------------------------------------------------------------

export class ApiError extends Error {
  status: number;
  errors: ValidationErrorItem[];

  constructor(status: number, message: string, errors: ValidationErrorItem[] = []) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.errors = errors;
  }
}

// ---------------------------------------------------------------------------
// Axios instance
// ---------------------------------------------------------------------------

const api: AxiosInstance = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000",
  timeout: 5000,
  headers: { "Content-Type": "application/json" },
});

api.interceptors.response.use(
  (response) => response,
  (error: AxiosError<{ detail?: ValidationErrorItem[] | string }>) => {
    const status = error.response?.status ?? 0;
    const data = error.response?.data;

    let message: string;
    let errors: ValidationErrorItem[] = [];

    if (data?.detail) {
      if (Array.isArray(data.detail)) {
        errors = data.detail;
        message = errors.map((e) => `${e.field}: ${e.message}`).join("; ");
      } else {
        message = String(data.detail);
      }
    } else {
      message = error.message || "Unknown API error";
    }

    console.error(`[API ${status}] ${message}`, { url: error.config?.url, errors });
    return Promise.reject(new ApiError(status, message, errors));
  },
);

// ---------------------------------------------------------------------------
// Typed API functions
// ---------------------------------------------------------------------------

export async function getModelInfo(): Promise<ModelMetadata> {
  const { data } = await api.get<ModelMetadata>("/api/v1/model/info");
  return data;
}

export async function postForecast(request: ForecastRequest): Promise<ForecastResponse> {
  const { data } = await api.post<ForecastResponse>("/api/v1/forecast/store", request);
  return data;
}

export async function postForecastDebug(request: ForecastRequest): Promise<ForecastDebugResponse> {
  const { data } = await api.post<ForecastDebugResponse>("/api/v1/forecast/store/debug", request);
  return data;
}

export async function postBacktest(request: BacktestRequest): Promise<BacktestResponse> {
  const { data } = await api.post<BacktestResponse>("/api/v1/backtest/store", request);
  return data;
}

export { api as default };
