import { useRef, useState } from "react";
import { ApiError, postCopilotExplain } from "../api/client";
import type { CopilotExplainResponse } from "../types/api";

const SUGGESTED = [
  "What is the current model health?",
  "Is there any feature drift?",
  "How accurate are the recent forecasts?",
  "When was the model last trained?",
];

function confidenceColor(c: number) {
  if (c >= 0.7) return "#16a34a";
  if (c >= 0.4) return "#d97706";
  return "#dc2626";
}

export default function CopilotPage() {
  const [query, setQuery] = useState(SUGGESTED[0]);
  const [response, setResponse] = useState<CopilotExplainResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await postCopilotExplain({ query: query.trim() });
      setResponse(data);
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : "Copilot request failed";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const useSuggestion = (s: string) => {
    setQuery(s);
    textareaRef.current?.focus();
  };

  return (
    <div style={{ padding: "1.5rem", maxWidth: 780 }}>
      <h1 style={{ fontSize: "1.4rem", fontWeight: 700, marginBottom: "0.5rem" }}>
        Copilot — Model Explainer
      </h1>
      <p style={{ fontSize: "0.85rem", color: "#6b7280", marginBottom: "1.2rem" }}>
        Ask a natural-language question about model health, drift, or forecast accuracy. The
        copilot explains results using monitoring data — it never generates predictions.
      </p>

      <div style={{ marginBottom: "0.75rem", display: "flex", flexWrap: "wrap", gap: 6 }}>
        {SUGGESTED.map((s) => (
          <button
            key={s}
            onClick={() => useSuggestion(s)}
            style={{
              padding: "3px 10px",
              fontSize: "0.78rem",
              borderRadius: 12,
              border: "1px solid #d1d5db",
              background: query === s ? "#eff6ff" : "#f9fafb",
              color: query === s ? "#2563eb" : "#374151",
              cursor: "pointer",
            }}
          >
            {s}
          </button>
        ))}
      </div>

      <form onSubmit={submit} style={{ marginBottom: "1.5rem" }}>
        <textarea
          ref={textareaRef}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          rows={3}
          style={{
            width: "100%",
            padding: "0.6rem 0.75rem",
            borderRadius: 6,
            border: "1px solid #d1d5db",
            fontSize: "0.9rem",
            resize: "vertical",
            boxSizing: "border-box",
          }}
          placeholder="Ask about model health, drift, forecast accuracy…"
        />
        <button
          type="submit"
          disabled={loading || !query.trim()}
          style={{
            marginTop: 8,
            padding: "0.5rem 1.4rem",
            borderRadius: 6,
            border: "none",
            background: loading || !query.trim() ? "#9ca3af" : "#2563eb",
            color: "#fff",
            fontWeight: 600,
            cursor: loading || !query.trim() ? "default" : "pointer",
          }}
        >
          {loading ? "Asking…" : "Ask Copilot"}
        </button>
      </form>

      {error && (
        <div
          style={{
            background: "#fee2e2",
            border: "1px solid #fca5a5",
            borderRadius: 8,
            padding: "0.75rem 1rem",
            color: "#991b1b",
            marginBottom: "1rem",
          }}
        >
          {error}
        </div>
      )}

      {response && (
        <div>
          <div style={{ marginBottom: "0.75rem", display: "flex", gap: 8, alignItems: "center" }}>
            <span
              style={{
                padding: "2px 10px",
                borderRadius: 12,
                fontSize: "0.76rem",
                fontWeight: 600,
                background: response.generator === "openai" ? "#eff6ff" : "#f3f4f6",
                color: response.generator === "openai" ? "#1d4ed8" : "#374151",
                border: "1px solid #e5e7eb",
              }}
            >
              {response.generator === "openai" ? "OpenAI" : "Rule-based"}
            </span>
            <span
              style={{
                padding: "2px 10px",
                borderRadius: 12,
                fontSize: "0.76rem",
                fontWeight: 600,
                background: `${confidenceColor(response.confidence)}18`,
                color: confidenceColor(response.confidence),
                border: `1px solid ${confidenceColor(response.confidence)}`,
              }}
            >
              Confidence: {Math.round(response.confidence * 100)}%
            </span>
          </div>

          <div
            style={{
              background: "#f0fdf4",
              border: "1px solid #bbf7d0",
              borderRadius: 8,
              padding: "1rem",
              marginBottom: "0.75rem",
            }}
          >
            <div style={{ fontWeight: 600, marginBottom: 6, color: "#166534" }}>Answer</div>
            <div style={{ fontSize: "0.9rem", lineHeight: 1.6, whiteSpace: "pre-wrap" }}>
              {response.answer}
            </div>
          </div>

          {response.reasoning && (
            <details style={{ marginBottom: "0.75rem" }}>
              <summary style={{ cursor: "pointer", fontSize: "0.85rem", color: "#6b7280", marginBottom: 4 }}>
                Reasoning
              </summary>
              <div
                style={{
                  background: "#f9fafb",
                  border: "1px solid #e5e7eb",
                  borderRadius: 6,
                  padding: "0.75rem",
                  fontSize: "0.85rem",
                  lineHeight: 1.6,
                  whiteSpace: "pre-wrap",
                }}
              >
                {response.reasoning}
              </div>
            </details>
          )}

          {response.intents?.length > 0 && (
            <div style={{ marginBottom: "0.75rem" }}>
              <span style={{ fontSize: "0.78rem", color: "#9ca3af" }}>Intents: </span>
              {response.intents.map((intent) => (
                <span
                  key={intent}
                  style={{
                    display: "inline-block",
                    padding: "1px 8px",
                    borderRadius: 10,
                    background: "#eff6ff",
                    color: "#1d4ed8",
                    fontSize: "0.76rem",
                    marginRight: 4,
                    border: "1px solid #bfdbfe",
                  }}
                >
                  {intent}
                </span>
              ))}
            </div>
          )}

          {response.sources?.length > 0 && (
            <details>
              <summary style={{ cursor: "pointer", fontSize: "0.78rem", color: "#9ca3af" }}>
                Sources ({response.sources.length})
              </summary>
              <ul style={{ fontSize: "0.78rem", color: "#6b7280", marginTop: 4, paddingLeft: 16 }}>
                {response.sources.map((s, i) => (
                  <li key={i}>
                    <strong>{s.type}</strong>
                    {s.note ? ` — ${s.note}` : ""}
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
