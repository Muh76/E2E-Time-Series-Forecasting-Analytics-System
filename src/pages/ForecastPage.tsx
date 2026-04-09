import { useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { postForecast, ApiError } from "../api/client";
import type { ForecastPoint } from "../types/api";

interface ChartRow {
  date: string;
  forecast: number;
  range: [number, number] | null;
}

function toChartData(points: ForecastPoint[]): ChartRow[] {
  return points.map((p) => ({
    date: p.date,
    forecast: p.forecast,
    range:
      p.confidence_low != null && p.confidence_high != null
        ? [p.confidence_low, p.confidence_high]
        : null,
  }));
}

function ForecastTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const row = payload[0]?.payload as ChartRow | undefined;
  if (!row) return null;

  return (
    <div style={{ background: "#fff", border: "1px solid #e5e7eb", borderRadius: 6, padding: "0.5rem 0.75rem", fontSize: "0.82rem" }}>
      <div style={{ fontWeight: 600, marginBottom: 4 }}>{label}</div>
      <div>Forecast: <strong>{row.forecast.toFixed(2)}</strong></div>
      {row.range && (
        <div style={{ color: "#6b7280" }}>
          95% CI: {row.range[0].toFixed(2)} – {row.range[1].toFixed(2)}
        </div>
      )}
    </div>
  );
}

export default function ForecastPage() {
  const [storeId, setStoreId] = useState(1);
  const [horizon, setHorizon] = useState(7);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<ChartRow[]>([]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setData([]);

    try {
      const res = await postForecast({ store_id: storeId, horizon });
      setData(toChartData(res.forecasts));
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.errors.length ? err.errors.map((e) => `${e.field}: ${e.message}`).join("\n") : err.message);
      } else {
        setError("Unexpected error");
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <h2>Store Forecast</h2>

      <form onSubmit={handleSubmit} style={{ display: "flex", gap: "1rem", alignItems: "flex-end", flexWrap: "wrap", marginBottom: "1.5rem" }}>
        <label style={{ display: "flex", flexDirection: "column", fontSize: "0.85rem" }}>
          Store ID
          <input type="number" min={1} value={storeId} onChange={(e) => setStoreId(Number(e.target.value))}
            style={{ marginTop: 4, padding: "0.4rem 0.6rem", border: "1px solid #d1d5db", borderRadius: 4, width: 100 }} />
        </label>
        <label style={{ display: "flex", flexDirection: "column", fontSize: "0.85rem" }}>
          Horizon
          <input type="number" min={1} max={60} value={horizon} onChange={(e) => setHorizon(Number(e.target.value))}
            style={{ marginTop: 4, padding: "0.4rem 0.6rem", border: "1px solid #d1d5db", borderRadius: 4, width: 80 }} />
        </label>
        <button type="submit" disabled={loading}
          style={{ padding: "0.45rem 1.2rem", background: loading ? "#9ca3af" : "#3b82f6", color: "#fff", border: "none", borderRadius: 4, cursor: loading ? "not-allowed" : "pointer", fontSize: "0.85rem" }}>
          {loading ? "Forecasting…" : "Generate"}
        </button>
      </form>

      {error && (
        <div style={{ padding: "0.75rem 1rem", background: "#fef2f2", border: "1px solid #fecaca", borderRadius: 6, color: "#991b1b", fontSize: "0.85rem", whiteSpace: "pre-wrap", marginBottom: "1rem" }}>
          {error}
        </div>
      )}

      {data.length > 0 && (
        <ResponsiveContainer width="100%" height={380}>
          <ComposedChart data={data} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="date" tick={{ fontSize: 12 }} angle={-30} textAnchor="end" height={60} />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip content={<ForecastTooltip />} />

            <Area
              dataKey="range"
              fill="#3b82f6"
              fillOpacity={0.12}
              stroke="none"
              isAnimationActive={false}
            />

            <Line
              dataKey="forecast"
              type="monotone"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={{ r: 3, fill: "#3b82f6" }}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      )}

      {data.length > 0 && (
        <table style={{ width: "100%", borderCollapse: "collapse", marginTop: "1rem", fontSize: "0.85rem" }}>
          <thead>
            <tr style={{ borderBottom: "2px solid #e5e7eb", textAlign: "left" }}>
              <th style={{ padding: "0.4rem 0.6rem" }}>Date</th>
              <th style={{ padding: "0.4rem 0.6rem" }}>Forecast</th>
              <th style={{ padding: "0.4rem 0.6rem" }}>Low</th>
              <th style={{ padding: "0.4rem 0.6rem" }}>High</th>
            </tr>
          </thead>
          <tbody>
            {data.map((row) => (
              <tr key={row.date} style={{ borderBottom: "1px solid #f3f4f6" }}>
                <td style={{ padding: "0.4rem 0.6rem" }}>{row.date}</td>
                <td style={{ padding: "0.4rem 0.6rem" }}>{row.forecast.toFixed(2)}</td>
                <td style={{ padding: "0.4rem 0.6rem" }}>{row.range?.[0]?.toFixed(2) ?? "—"}</td>
                <td style={{ padding: "0.4rem 0.6rem" }}>{row.range?.[1]?.toFixed(2) ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
