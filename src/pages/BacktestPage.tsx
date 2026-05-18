import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { postBacktest, ApiError } from "../api/client";
import type { BacktestResponse, BacktestSplit } from "../types/api";

const METRIC_COLORS: Record<string, string> = {
  rmse: "#3b82f6",
  mae: "#10b981",
  mape: "#f59e0b",
};

interface ChartRow {
  split: string;
  rmse: number;
  mae: number;
  mape: number;
}

function toChartData(splits: BacktestSplit[]): ChartRow[] {
  return splits.map((s) => ({
    split: `Split ${s.split}`,
    rmse: parseFloat(s.rmse.toFixed(2)),
    mae: parseFloat(s.mae.toFixed(2)),
    mape: parseFloat((s.mape * 100).toFixed(2)),
  }));
}

function MetricBadge({ label, value, unit = "" }: { label: string; value: number; unit?: string }) {
  return (
    <div style={{
      flex: "1 1 120px",
      padding: "0.75rem 1rem",
      background: "#f9fafb",
      border: "1px solid #e5e7eb",
      borderRadius: 8,
      textAlign: "center",
    }}>
      <div style={{ fontSize: "0.75rem", color: "#6b7280", marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: "1.25rem", fontWeight: 700, color: "#1f2937" }}>
        {value.toFixed(2)}{unit}
      </div>
    </div>
  );
}

export default function BacktestPage() {
  const [storeId, setStoreId] = useState(1);
  const [horizon, setHorizon] = useState(7);
  const [nSplits, setNSplits] = useState(3);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<BacktestResponse | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await postBacktest({ store_id: storeId, horizon, n_splits: nSplits });
      setResult(res);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(
          err.errors.length
            ? err.errors.map((e) => `${e.field}: ${e.message}`).join("\n")
            : err.message,
        );
      } else {
        setError("Unexpected error");
      }
    } finally {
      setLoading(false);
    }
  }

  const chartData = result ? toChartData(result.splits) : [];

  return (
    <div>
      <h2 style={{ marginTop: 0, marginBottom: "1.25rem" }}>Rolling-Origin Backtest</h2>

      {/* Form */}
      <form onSubmit={handleSubmit} style={{ display: "flex", flexWrap: "wrap", gap: "0.75rem", alignItems: "flex-end", marginBottom: "1.5rem" }}>
        <label style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: "0.85rem", color: "#374151" }}>
          Store ID
          <input
            type="number"
            min={1}
            max={1115}
            value={storeId}
            onChange={(e) => setStoreId(Number(e.target.value))}
            style={{ padding: "0.4rem 0.6rem", border: "1px solid #d1d5db", borderRadius: 6, width: 90, fontSize: "0.9rem" }}
            required
          />
        </label>

        <label style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: "0.85rem", color: "#374151" }}>
          Horizon (days)
          <input
            type="number"
            min={1}
            max={60}
            value={horizon}
            onChange={(e) => setHorizon(Number(e.target.value))}
            style={{ padding: "0.4rem 0.6rem", border: "1px solid #d1d5db", borderRadius: 6, width: 90, fontSize: "0.9rem" }}
            required
          />
        </label>

        <label style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: "0.85rem", color: "#374151" }}>
          Splits
          <input
            type="number"
            min={1}
            max={20}
            value={nSplits}
            onChange={(e) => setNSplits(Number(e.target.value))}
            style={{ padding: "0.4rem 0.6rem", border: "1px solid #d1d5db", borderRadius: 6, width: 90, fontSize: "0.9rem" }}
            required
          />
        </label>

        <button
          type="submit"
          disabled={loading}
          style={{
            padding: "0.45rem 1.2rem",
            background: loading ? "#93c5fd" : "#3b82f6",
            color: "#fff",
            border: "none",
            borderRadius: 6,
            fontSize: "0.9rem",
            cursor: loading ? "not-allowed" : "pointer",
            fontWeight: 500,
          }}
        >
          {loading ? "Running…" : "Run Backtest"}
        </button>
      </form>

      {/* Error */}
      {error && (
        <div style={{
          padding: "0.75rem 1rem",
          background: "#fef2f2",
          border: "1px solid #fecaca",
          borderRadius: 6,
          color: "#991b1b",
          fontSize: "0.85rem",
          whiteSpace: "pre-wrap",
          marginBottom: "1rem",
        }}>
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <>
          {/* Average metrics */}
          <div style={{ marginBottom: "1.5rem" }}>
            <h3 style={{ fontSize: "0.95rem", color: "#6b7280", marginBottom: "0.6rem", fontWeight: 500 }}>
              Average across {result.n_splits} splits — horizon {result.horizon}d
            </h3>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.75rem" }}>
              <MetricBadge label="Avg RMSE" value={result.average.rmse} />
              <MetricBadge label="Avg MAE" value={result.average.mae} />
              <MetricBadge label="Avg MAPE" value={result.average.mape * 100} unit="%" />
            </div>
          </div>

          {/* Bar chart */}
          <div style={{ marginBottom: "1.5rem" }}>
            <h3 style={{ fontSize: "0.95rem", color: "#6b7280", marginBottom: "0.6rem", fontWeight: 500 }}>
              Metrics per split (MAPE as %)
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="split" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip
                  formatter={(value: number, name: string) =>
                    name === "mape" ? [`${value.toFixed(2)}%`, "MAPE"] : [value.toFixed(2), name.toUpperCase()]
                  }
                  contentStyle={{ fontSize: "0.82rem", borderRadius: 6 }}
                />
                <Legend formatter={(v) => v === "mape" ? "MAPE (%)" : v.toUpperCase()} />
                <Bar dataKey="rmse" fill={METRIC_COLORS.rmse} radius={[3, 3, 0, 0]}>
                  {chartData.map((_, i) => <Cell key={i} fill={METRIC_COLORS.rmse} />)}
                </Bar>
                <Bar dataKey="mae" fill={METRIC_COLORS.mae} radius={[3, 3, 0, 0]}>
                  {chartData.map((_, i) => <Cell key={i} fill={METRIC_COLORS.mae} />)}
                </Bar>
                <Bar dataKey="mape" fill={METRIC_COLORS.mape} radius={[3, 3, 0, 0]}>
                  {chartData.map((_, i) => <Cell key={i} fill={METRIC_COLORS.mape} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Per-split table */}
          <div>
            <h3 style={{ fontSize: "0.95rem", color: "#6b7280", marginBottom: "0.6rem", fontWeight: 500 }}>
              Split details
            </h3>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem" }}>
              <thead>
                <tr style={{ borderBottom: "2px solid #e5e7eb", textAlign: "left" }}>
                  <th style={{ padding: "0.4rem 0.6rem" }}>Split</th>
                  <th style={{ padding: "0.4rem 0.6rem" }}>Cutoff Date</th>
                  <th style={{ padding: "0.4rem 0.6rem" }}>Horizon</th>
                  <th style={{ padding: "0.4rem 0.6rem" }}>RMSE</th>
                  <th style={{ padding: "0.4rem 0.6rem" }}>MAE</th>
                  <th style={{ padding: "0.4rem 0.6rem" }}>MAPE</th>
                </tr>
              </thead>
              <tbody>
                {result.splits.map((s) => (
                  <tr key={s.split} style={{ borderBottom: "1px solid #f3f4f6" }}>
                    <td style={{ padding: "0.4rem 0.6rem", fontWeight: 500 }}>{s.split}</td>
                    <td style={{ padding: "0.4rem 0.6rem", color: "#6b7280" }}>{s.cutoff_date}</td>
                    <td style={{ padding: "0.4rem 0.6rem" }}>{s.horizon}d</td>
                    <td style={{ padding: "0.4rem 0.6rem" }}>{s.rmse.toFixed(2)}</td>
                    <td style={{ padding: "0.4rem 0.6rem" }}>{s.mae.toFixed(2)}</td>
                    <td style={{ padding: "0.4rem 0.6rem" }}>{(s.mape * 100).toFixed(2)}%</td>
                  </tr>
                ))}
                <tr style={{ borderTop: "2px solid #e5e7eb", background: "#f9fafb", fontWeight: 600 }}>
                  <td colSpan={3} style={{ padding: "0.4rem 0.6rem" }}>Average</td>
                  <td style={{ padding: "0.4rem 0.6rem" }}>{result.average.rmse.toFixed(2)}</td>
                  <td style={{ padding: "0.4rem 0.6rem" }}>{result.average.mae.toFixed(2)}</td>
                  <td style={{ padding: "0.4rem 0.6rem" }}>{(result.average.mape * 100).toFixed(2)}%</td>
                </tr>
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
