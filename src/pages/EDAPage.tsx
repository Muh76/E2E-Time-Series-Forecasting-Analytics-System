import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ApiError, getModelInfo } from "../api/client";
import type { FeatureImportanceItem, ModelMetadata } from "../types/api";

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section style={{ marginBottom: "1.75rem" }}>
      <h2 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: 10, borderBottom: "1px solid #e5e7eb", paddingBottom: 4 }}>
        {title}
      </h2>
      {children}
    </section>
  );
}

function KV({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div style={{ display: "flex", gap: 8, fontSize: "0.85rem", marginBottom: 4 }}>
      <span style={{ color: "#6b7280", minWidth: 160 }}>{label}</span>
      <span style={{ color: "#111827", fontWeight: 500 }}>{value ?? "—"}</span>
    </div>
  );
}

function ImportanceChart({ items }: { items: FeatureImportanceItem[] }) {
  const top20 = [...items]
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 20);
  const max = top20[0]?.importance ?? 1;

  return (
    <ResponsiveContainer width="100%" height={Math.max(200, top20.length * 22)}>
      <BarChart data={top20} layout="vertical" margin={{ left: 8, right: 24, top: 4, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" horizontal={false} />
        <XAxis type="number" tick={{ fontSize: 11 }} domain={[0, max]} />
        <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={130} />
        <Tooltip
          formatter={(v: number) => v.toFixed(4)}
          contentStyle={{ fontSize: "0.8rem" }}
        />
        <Bar dataKey="importance" radius={[0, 3, 3, 0]}>
          {top20.map((_, i) => (
            <Cell
              key={i}
              fill={i === 0 ? "#2563eb" : i < 5 ? "#3b82f6" : "#93c5fd"}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export default function EDAPage() {
  const [meta, setMeta] = useState<ModelMetadata | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getModelInfo()
      .then((data) => { setMeta(data); })
      .catch((e) => {
        const msg = e instanceof ApiError ? e.message : "Failed to load model info";
        setError(msg);
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <div style={{ padding: "2rem", color: "#6b7280" }}>Loading model info…</div>;
  }

  if (error || !meta) {
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
          <strong>Model info unavailable:</strong> {error ?? "No data returned"}
          <p style={{ fontSize: "0.82rem", marginTop: 6 }}>
            Run <code>python scripts/train.py</code> to generate artifacts, then restart the API.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div style={{ padding: "1.5rem", maxWidth: 900 }}>
      <h1 style={{ fontSize: "1.4rem", fontWeight: 700, marginBottom: "1.5rem" }}>
        Exploratory Data Analysis
      </h1>

      <Section title="Model Overview">
        <KV label="Version" value={meta.model_version} />
        <KV label="Trained at" value={meta.trained_at} />
        <KV
          label="Training period"
          value={
            meta.training_date_range
              ? `${meta.training_date_range.start} → ${meta.training_date_range.end}`
              : undefined
          }
        />
        <KV label="Training rows" value={meta.sample_size?.toLocaleString()} />
        <KV label="Feature count" value={meta.feature_count} />
        <KV label="Max lag" value={meta.max_lag} />
        <KV label="Lookback window" value={meta.lookback_window} />
        <KV label="Residual std" value={meta.residual_std?.toFixed(4)} />
      </Section>

      <Section title="Validation Metrics">
        {meta.validation_metrics ? (
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            {(["rmse", "mae", "mape"] as const).map((k) => (
              <div
                key={k}
                style={{
                  background: "#f9fafb",
                  border: "1px solid #e5e7eb",
                  borderRadius: 8,
                  padding: "0.65rem 1rem",
                  minWidth: 100,
                  textAlign: "center",
                }}
              >
                <div style={{ fontSize: "0.72rem", color: "#6b7280", marginBottom: 3 }}>
                  {k.toUpperCase()}
                </div>
                <div style={{ fontSize: "1.25rem", fontWeight: 700 }}>
                  {meta.validation_metrics[k] !== null
                    ? (meta.validation_metrics[k] as number).toFixed(2)
                    : "—"}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p style={{ color: "#6b7280", fontSize: "0.85rem" }}>No validation metrics available.</p>
        )}
      </Section>

      {meta.feature_importance?.length > 0 && (
        <Section title={`Feature Importance (top ${Math.min(20, meta.feature_importance.length)} of ${meta.feature_importance.length})`}>
          <ImportanceChart items={meta.feature_importance} />
        </Section>
      )}

      {meta.feature_columns?.length > 0 && (
        <Section title={`Feature Columns (${meta.feature_columns.length})`}>
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: 6,
              maxHeight: 200,
              overflowY: "auto",
            }}
          >
            {meta.feature_columns.map((col) => (
              <span
                key={col}
                style={{
                  padding: "2px 8px",
                  borderRadius: 10,
                  background: "#f3f4f6",
                  border: "1px solid #e5e7eb",
                  fontSize: "0.75rem",
                  color: "#374151",
                }}
              >
                {col}
              </span>
            ))}
          </div>
        </Section>
      )}
    </div>
  );
}
