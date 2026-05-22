import { useEffect, useRef, useState } from "react";
import { ApiError, postChatQuery } from "../api/client";
import type { ChatSource } from "../types/api";

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: ChatSource[];
  error?: boolean;
}

function UserBubble({ text }: { text: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 10 }}>
      <div
        style={{
          background: "#2563eb",
          color: "#fff",
          borderRadius: "12px 12px 2px 12px",
          padding: "0.55rem 0.9rem",
          maxWidth: "70%",
          fontSize: "0.88rem",
          lineHeight: 1.5,
        }}
      >
        {text}
      </div>
    </div>
  );
}

function AssistantBubble({ content, sources, error }: { content: string; sources?: ChatSource[]; error?: boolean }) {
  return (
    <div style={{ display: "flex", justifyContent: "flex-start", marginBottom: 10 }}>
      <div
        style={{
          background: error ? "#fee2e2" : "#f9fafb",
          border: `1px solid ${error ? "#fca5a5" : "#e5e7eb"}`,
          color: error ? "#991b1b" : "#111827",
          borderRadius: "12px 12px 12px 2px",
          padding: "0.55rem 0.9rem",
          maxWidth: "78%",
          fontSize: "0.88rem",
          lineHeight: 1.6,
          whiteSpace: "pre-wrap",
        }}
      >
        {content}
        {sources && sources.length > 0 && (
          <details style={{ marginTop: 6 }}>
            <summary style={{ cursor: "pointer", fontSize: "0.73rem", color: "#9ca3af" }}>
              {sources.length} source{sources.length !== 1 ? "s" : ""}
            </summary>
            <ul style={{ margin: "4px 0 0", paddingLeft: 14, fontSize: "0.73rem", color: "#6b7280" }}>
              {sources.map((s, i) => (
                <li key={i}>
                  <strong>{s.source}</strong>: {s.header}{" "}
                  <span style={{ color: "#9ca3af" }}>(score: {s.score.toFixed(3)})</span>
                </li>
              ))}
            </ul>
          </details>
        )}
      </div>
    </div>
  );
}

const WELCOME = "Hi! Ask me anything about this forecasting system — architecture, API endpoints, feature engineering, training, or deployment.";

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: WELCOME },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setLoading(true);
    try {
      const resp = await postChatQuery({ message: text });
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: resp.reply, sources: resp.sources },
      ]);
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : "Chat request failed";
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: msg, error: true },
      ]);
    } finally {
      setLoading(false);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  };

  const handleKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "calc(100vh - 140px)",
        maxWidth: 780,
        padding: "1rem 1.5rem",
      }}
    >
      <h1 style={{ fontSize: "1.4rem", fontWeight: 700, marginBottom: "0.25rem" }}>Chat</h1>
      <p style={{ fontSize: "0.82rem", color: "#6b7280", marginBottom: "0.75rem" }}>
        RAG-backed Q&A over the system documentation. Powered by the FAISS index built from the
        project's docs and README.
      </p>

      <div
        style={{
          flex: 1,
          overflowY: "auto",
          border: "1px solid #e5e7eb",
          borderRadius: 10,
          padding: "0.75rem",
          background: "#fff",
          marginBottom: "0.75rem",
        }}
      >
        {messages.map((m, i) =>
          m.role === "user" ? (
            <UserBubble key={i} text={m.content} />
          ) : (
            <AssistantBubble key={i} content={m.content} sources={m.sources} error={m.error} />
          )
        )}
        {loading && (
          <AssistantBubble content="…" />
        )}
        <div ref={bottomRef} />
      </div>

      <div style={{ display: "flex", gap: 8 }}>
        <textarea
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKey}
          rows={2}
          disabled={loading}
          style={{
            flex: 1,
            padding: "0.55rem 0.75rem",
            borderRadius: 8,
            border: "1px solid #d1d5db",
            fontSize: "0.88rem",
            resize: "none",
            outline: "none",
          }}
          placeholder="Ask about the system… (Enter to send, Shift+Enter for newline)"
        />
        <button
          onClick={send}
          disabled={loading || !input.trim()}
          style={{
            padding: "0 1.1rem",
            borderRadius: 8,
            border: "none",
            background: loading || !input.trim() ? "#9ca3af" : "#2563eb",
            color: "#fff",
            fontWeight: 600,
            fontSize: "0.88rem",
            cursor: loading || !input.trim() ? "default" : "pointer",
            alignSelf: "stretch",
          }}
        >
          Send
        </button>
      </div>
    </div>
  );
}
