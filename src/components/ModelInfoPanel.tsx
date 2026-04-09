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

  const vm = meta.validation_metrics;

  return (
    <div className="model-info-panel">
      <span><strong>Model</strong> {meta.model_version}</span>
      <span><strong>Features</strong> {meta.feature_count}</span>
      <span><strong>Samples</strong> {meta.sample_size.toLocaleString()}</span>
      <span><strong>Train</strong> {meta.training_date_range.start} → {meta.training_date_range.end}</span>
      {vm.rmse != null && <span><strong>RMSE</strong> {vm.rmse.toFixed(2)}</span>}
      {vm.mae != null && <span><strong>MAE</strong> {vm.mae.toFixed(2)}</span>}
    </div>
  );
}
