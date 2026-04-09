import { useEffect, useState } from "react";
import apiClient from "../api/client";

type Status = "loading" | "ok" | "error";

export default function HealthDot() {
  const [status, setStatus] = useState<Status>("loading");

  useEffect(() => {
    let mounted = true;

    const check = () =>
      apiClient
        .get("/health/live", { timeout: 3000 })
        .then(() => mounted && setStatus("ok"))
        .catch(() => mounted && setStatus("error"));

    check();
    const id = setInterval(check, 30_000);
    return () => { mounted = false; clearInterval(id); };
  }, []);

  const color = status === "ok" ? "#22c55e" : status === "error" ? "#ef4444" : "#9ca3af";
  const title = status === "ok" ? "API healthy" : status === "error" ? "API unreachable" : "Checking…";

  return (
    <span
      className="health-dot"
      title={title}
      style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: color }}
    />
  );
}
