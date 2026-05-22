import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ApiError, getMonitoringSummary } from "../api/client";
import type { MonitoringSummary } from "../types/api";

const STATUS_COLOR: Record<string, string> = {
  healthy: "#16a34a",
  warning: "#d97706",
  unknown: "#6b7280",
};

function alertColor(active: boolean) {
  return active ? "#dc2626" : "#16a34a";
}

function Badge({ label, active }: { label: string; active: boolean }) {
  const color = alertColor(active);
  return (
    <span
      style={{
        display: "inline-block",
        padding: "2px 10px",
        borderRadius: 12,
        fontSize: "0.78rem",
        fontWeight: 600,
        background: active ? "#fee2e2" : "#dcfce7",
        color,
        border: `1px solid ${color}`,
        marginRight: 6,
      }}
    >
      {label}: {active ? "ALERT" : "ok"}
    </span>
  );
}

function MetricCard({ label, value, unit = "" }: { label: string; value: number | null; unit?: string }) {
  return (
    <div
      style={{
        background: "#f9fafb",
        border: "1px solid #e5e7eb",
        borderRadius: 8,
        padding: "0.75rem 1rem",
        minWidth: 120,
        textAlign: "center",
      }}
    >
      <div style={{ fontSize: "0.75rem", color: "#6b7280", marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: "1.4rem", fontWeight: 700, color: "#111827" }}>
        {value !== null && value !== undefined ? value.toFixed(2) : "—"}
        {unit && <span style={{ fontSize: "0.85rem", marginLeft: 2 }}>{unit}</span>}
      </div>
    </div>
  );
}

export default function MonitoringPage() {
  const [summary, setSummary] = useState<MonitoringSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

  const refresh = async () => {
    try {
      const data = await getMonitoringSummary();
      setSummary(data);
      setError(null);
      setLastRefresh(new Date());
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : "Failed to load monitoring data";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 30_000);
    return () => clearInterval(timer);
  }, []);

  if (loading) {
    return <div style={{ padding: "2rem", color: "#6b7280" }}>Loading monitoring data…</div>;
  }

  if (error && !summary) {
    return (
      <div style={{ padding: "2rem" }}>
        <div
          style={{
            background: "#fee2e2",
            border: "1px solid #fca5a5",
            borderRadius: 8,
            padding: "1rem",
            color: "#991b1b",
          }}
        >
          <strong>Monitoring unavailable:</strong> {error}
        </div>
      </div>
    );
  }

  if (!summary) return null;

  const statusColor = STATUS_COLOR[summary.overall_status] ?? "#6b7280";
  const rollingMae = summary.rolling_series?.mae ?? [];

  return (
    <div style={{ padding: "1.5rem", maxWidth: 900 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: "1.5rem" }}>
        <h1 style={{ margin: 0, fontSize: "1.4rem", fontWeight: 700 }}>Model Monitoring</h1>
        <span
          style={{
            padding: "3px 14px",
            borderRadius: 12,
            fontWeight: 700,
            fontSize: "0.85rem",
            background: `${statusColor}20`,
            color: statusColor,
            border: `1px solid ${statusColor}`,
            textTransform: "capitalize",
          }}
        >
          {summary.overall_status}
        </span>
        {lastRefresh && (
          <span style={{ marginLeft: "auto", fontSize: "0.75rem", color: "#9ca3af" }}>
            Refreshed {lastRefresh.toLocaleTimeString()}
          </span>
        )}
      </div>

      <section style={{ marginBottom: "1.5rem" }}>
        <h2 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: 8 }}>Alerts</h2>
        <Badge label="MAE" active={summary.alerts.mae} />
        <Badge label="MAPE" active={summary.alerts.mape} />
        <Badge label="Drift" active={summary.alerts.drift} />
      </section>

      <section style={{ marginBottom: "1.5rem" }}>
        <h2 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: 8 }}>Performance</h2>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          <MetricCard label="MAE" value={summary.performance.mae} />
          <MetricCard label="RMSE" value={summary.performance.rmse} />
          <MetricCard label="MAPE" value={summary.performance.mape} unit="%" />
          <MetricCard label="Sample size" value={summary.performance.sample_size} />
        </div>
        {summary.performance.source && (
          <p style={{ fontSize: "0.78rem", color: "#9ca3af", marginTop: 6 }}>
            Source: {summary.performance.source}
          </p>
        )}
      </section>

      {rollingMae.length > 0 && (
        <section style={{ marginBottom: "1.5rem" }}>
          <h2 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: 8 }}>Rolling MAE</h2>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={rollingMae}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Line type="monotone" dataKey="value" stroke="#3b82f6" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </section>
      )}

      <section style={{ marginBottom: "1.5rem" }}>
        <h2 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: 8 }}>Feature Drift</h2>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 6 }}>
          <span
            style={{
              padding: "2px 10px",
              borderRadius: 12,
              fontSize: "0.8rem",
              fontWeight: 600,
              background: summary.drift.status === "drift_detected" ? "#fee2e2" : "#dcfce7",
              color: summary.drift.status === "drift_detected" ? "#991b1b" : "#166534",
            }}
          >
            {summary.drift.status === "drift_detected" ? "Drift detected" : "No drift"}
          </span>
          {summary.drift.overall_score !== null && (
            <span style={{ fontSize: "0.82rem", color: "#6b7280" }}>
              Score: {summary.drift.overall_score.toFixed(3)} (threshold: {summary.drift.threshold})
            </span>
          )}
        </div>
        {summary.drift.indicators.length > 0 && (
          <table style={{ fontSize: "0.82rem", borderCollapse: "collapse", width: "100%" }}>
            <thead>
              <tr style={{ background: "#f9fafb" }}>
                <th style={{ padding: "4px 8px", textAlign: "left", border: "1px solid #e5e7eb" }}>Feature</th>
                <th style={{ padding: "4px 8px", textAlign: "right", border: "1px solid #e5e7eb" }}>Score</th>
              </tr>
            </thead>
            <tbody>
              {summary.drift.indicators.map((ind) => (
                <tr key={ind.feature}>
                  <td style={{ padding: "4px 8px", border: "1px solid #e5e7eb" }}>{ind.feature}</td>
                  <td style={{ padding: "4px 8px", textAlign: "right", border: "1px solid #e5e7eb" }}>
                    {ind.score.toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section>
        <h2 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: 8 }}>Pipeline</h2>
        <div style={{ fontSize: "0.85rem", color: "#374151", lineHeight: 1.8 }}>
          <div>
            <strong>Status:</strong>{" "}
            <span style={{ textTransform: "capitalize" }}>{summary.pipeline.status}</span>
          </div>
          <div><strong>Last training:</strong> {summary.pipeline.last_training ?? "—"}</div>
          <div><strong>Last ETL:</strong> {summary.pipeline.last_etl ?? "—"}</div>
          <div><strong>Model version:</strong> {summary.model_version}</div>
        </div>
      </section>
    </div>
  );
}
