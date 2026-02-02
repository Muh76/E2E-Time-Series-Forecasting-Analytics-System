"""
Data drift detection for numeric features.

Compares reference (training) vs current distribution using a simple PSI-like
score: bin values, compare proportions, aggregate absolute differences.
No SciPy; no file I/O; deterministic.
"""

from typing import Any

import numpy as np
import pandas as pd


class DataDriftDetector:
    """
    Detect distribution drift between reference and current data.

    Compares reference (training) distribution vs current distribution.
    Supports numeric features only. Implements a simple PSI-like score:
    bin values, compare proportions per bin, aggregate absolute differences.
    Returns per-feature drift scores and an overall score; flags drift if
    score exceeds config threshold.

    **Assumptions:**
    - Numeric features only; non-numeric columns are ignored.
    - Reference and current must have the same feature columns (or a subset).
    - Bin edges are derived from the reference distribution so that current
      is evaluated against a fixed baseline. This ensures deterministic,
      comparable scores across runs.
    - Bins use equal-width boundaries from reference min/max. Bins with zero
      reference count get a small epsilon (1e-10) to avoid division issues;
      the score uses proportion differences only (no log), so no division.
    - Minimum sample size (from config) is a soft check: if below, the
      detector still runs but results may be unreliable; document for callers.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Initialize with drift config.

        Config shape (from monitoring.drift):
            threshold: float, flag drift if overall score > this (default 0.25)
            min_sample_size: int, soft minimum for reliable scores (default 100)
            n_bins: int, number of bins for histogram (default 10)
        """
        cfg = (config or {}).get("monitoring", {}).get("drift", {}) or config or {}
        self._threshold = float(cfg.get("threshold", 0.25))
        self._min_sample_size = int(cfg.get("min_sample_size", 100))
        self._n_bins = int(cfg.get("n_bins", 10))
        self._ref_data: pd.DataFrame | None = None
        self._bin_edges: dict[str, np.ndarray] = {}

    def fit_reference(self, reference_df: pd.DataFrame, feature_cols: list[str] | None = None) -> "DataDriftDetector":
        """
        Fit on reference (training) data. Stores distribution and bin edges.

        Args:
            reference_df: Reference DataFrame (e.g. training features).
            feature_cols: Columns to use; if None, use all numeric columns.

        Returns:
            self, for chaining.
        """
        if feature_cols is None:
            feature_cols = [
                c for c in reference_df.columns
                if pd.api.types.is_numeric_dtype(reference_df[c])
            ]
        self._ref_data = reference_df[feature_cols].copy()
        self._bin_edges = {}
        for col in feature_cols:
            vals = self._ref_data[col].dropna().values
            if len(vals) == 0:
                self._bin_edges[col] = np.array([0.0, 1.0])
            elif len(vals) == 1:
                v = float(vals[0])
                self._bin_edges[col] = np.array([v, v + 1e-10])
            else:
                edges = np.linspace(np.nanmin(vals), np.nanmax(vals), self._n_bins + 1)
                # Ensure right edge is strictly greater so all values fall in [left, right)
                edges[-1] = edges[-1] + 1e-10
                self._bin_edges[col] = edges
        return self

    def detect_drift(
        self,
        current_df: pd.DataFrame,
        feature_cols: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Compare current distribution to reference. Return drift scores and flag.

        Args:
            current_df: Current DataFrame (e.g. production features).
            feature_cols: Columns to use; if None, use columns from fit_reference.

        Returns:
            Dictionary with:
                - per_feature_scores: dict[str, float] drift score per feature
                - overall_score: float (mean of per-feature scores)
                - drift_detected: bool (overall_score > threshold)
                - threshold: float
                - n_bins: int
                - ref_sample_size: int
                - current_sample_size: int
        """
        if self._ref_data is None:
            raise RuntimeError("DataDriftDetector must call fit_reference before detect_drift.")
        cols = feature_cols or list(self._ref_data.columns)
        for c in cols:
            if c not in self._bin_edges:
                raise ValueError(f"Feature '{c}' not in reference; call fit_reference with matching columns.")
        ref_data = self._ref_data[cols]
        current_data = current_df[cols].copy()

        per_feature_scores: dict[str, float] = {}
        for col in cols:
            ref_vals = ref_data[col].dropna().values
            cur_vals = current_data[col].dropna().values
            edges = self._bin_edges[col]
            # Histogram counts; use same edges for both
            ref_hist, _ = np.histogram(ref_vals, bins=edges)
            cur_hist, _ = np.histogram(cur_vals, bins=edges)
            ref_prop = ref_hist / (ref_hist.sum() + 1e-10)
            cur_prop = cur_hist / (cur_hist.sum() + 1e-10)
            # PSI-like: aggregate absolute differences of proportions
            score = float(np.sum(np.abs(cur_prop - ref_prop)))
            per_feature_scores[col] = score

        overall_score = float(np.mean(list(per_feature_scores.values()))) if per_feature_scores else 0.0
        drift_detected = overall_score > self._threshold

        return {
            "per_feature_scores": per_feature_scores,
            "overall_score": overall_score,
            "drift_detected": drift_detected,
            "threshold": self._threshold,
            "n_bins": self._n_bins,
            "ref_sample_size": len(ref_data),
            "current_sample_size": len(current_data),
        }
