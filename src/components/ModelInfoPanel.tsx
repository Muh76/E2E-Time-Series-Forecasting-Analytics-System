import { useEffect, useState } from "react";
import { getModelInfo } from "../api/client";
import type { ModelMetadata } from "../types/api";

export default function ModelInfoPanel() {
  const [meta, setMeta] = useState<ModelMetadata | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getModelInfo()
      .then(setMeta)
      .catch((err) => setError(err.message ?? "Failed to load model info"));
  }, []);

  if (error) {
    return <div className="model-info-panel model-info-error">{error}</div>;
  }
  if (!meta) {
    return <div className="model-info-panel">Loading model info…</div>;
  }

  return (
    <div className="model-info-panel">
      <span><strong>Model</strong> {meta.model_version}</span>
      <span><strong>Features</strong> {meta.feature_count}</span>
      <span><strong>Train</strong> {meta.train_start} → {meta.train_end}</span>
      <span><strong>RMSE</strong> {meta.train_rmse.toFixed(2)}</span>
      <span><strong>MAE</strong> {meta.train_mae.toFixed(2)}</span>
    </div>
  );
}
