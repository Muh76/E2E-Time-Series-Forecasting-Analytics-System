"""
Input distribution drift vs training window (processed data).

Compares the first chronological half (training reference) to the second half
(current) using normalized mean and standard deviation differences, optionally
blended with a two-sample KS statistic when SciPy is installed.

Returns drift_score in [0, 1] and status low | medium | high.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from backend.app.runtime_paths import base_default_config_path, processed_parquet_path

logger = logging.getLogger(__name__)

try:
    from scipy import stats as scipy_stats

    _HAS_SCIPY = True
except ImportError:
    scipy_stats = None  # type: ignore[assignment]
    _HAS_SCIPY = False

_PREFERRED_NUMERIC = (
    "target_cleaned",
    "Sales",
    "Customers",
    "Open",
    "Promo",
    "SchoolHoliday",
)

_EPS = 1e-9
_MIN_SAMPLES_PER_WINDOW = 20


def _load_target_column(config_path: Path) -> str:
    if not config_path.exists():
        return "target_cleaned"
    with config_path.open() as f:
        cfg = yaml.safe_load(f) or {}
    fe = cfg.get("feature_engineering") or {}
    return str(fe.get("target_column", "target_cleaned"))


def _pick_numeric_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    cols: list[str] = []
    for c in _PREFERRED_NUMERIC:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    if target_col in df.columns and target_col not in cols and pd.api.types.is_numeric_dtype(df[target_col]):
        cols.insert(0, target_col)
    if not cols:
        cols = [
            c
            for c in df.columns
            if c not in ("store_id", "date", "Date", "Store") and pd.api.types.is_numeric_dtype(df[c])
        ][:12]
    return cols


def _normalize_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _feature_drift_components(
    ref: np.ndarray,
    cur: np.ndarray,
) -> tuple[float, float | None]:
    """
    Return (mean_std_score in [0,1], ks_statistic in [0,1] or None).
    """
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < 2 or len(cur) < 2:
        return 0.0, None

    m_r, s_r = float(np.mean(ref)), float(np.std(ref, ddof=1))
    m_c, s_c = float(np.mean(cur)), float(np.std(cur, ddof=1))

    mean_norm = abs(m_c - m_r) / (abs(m_r) + _EPS)
    mean_norm = min(mean_norm, 1.0)

    std_norm = abs(s_c - s_r) / (s_r + _EPS)
    std_norm = min(std_norm, 1.0)

    score_ms = (mean_norm + std_norm) / 2.0

    ks_val: float | None = None
    if _HAS_SCIPY and scipy_stats is not None:
        try:
            res = scipy_stats.ks_2samp(ref, cur)
            ks_val = float(min(max(res.statistic, 0.0), 1.0))
        except Exception as exc:
            logger.debug("KS test skipped: %s", exc)

    return score_ms, ks_val


def compute_distribution_drift(
    parquet_path: Path | None = None,
    config_path: Path | None = None,
    store_id: int | None = None,
) -> dict[str, Any] | None:
    """
    Compare early (training) vs late (current) windows on processed data.

    Args:
        parquet_path: Defaults to ``data/processed/etl_output.parquet`` under project root.
        config_path: Defaults to ``config/base/default.yaml`` (target column).
        store_id: If set, restrict to this store; else use all rows.

    Returns:
        ``{"drift_score": float, "status": "low"|"medium"|"high"}`` plus optional
        ``ks_used`` (bool) and ``n_features`` (int), or ``None`` if data is insufficient.
    """
    pq = parquet_path or processed_parquet_path()
    cfg_path = config_path or base_default_config_path()

    if not pq.exists():
        logger.warning("Drift: parquet not found at %s", pq)
        return None

    target_col = _load_target_column(cfg_path)
    try:
        df = pd.read_parquet(pq)
    except Exception as exc:
        logger.warning("Drift: failed to read parquet: %s", exc)
        return None

    if store_id is not None and "store_id" in df.columns:
        df = df[df["store_id"] == int(store_id)].copy()
    if df.empty:
        return None

    if "date" in df.columns:
        df["_sort_date"] = _normalize_date_series(df["date"])
        df = df.sort_values("_sort_date").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    n = len(df)
    split = n // 2
    if split < _MIN_SAMPLES_PER_WINDOW or (n - split) < _MIN_SAMPLES_PER_WINDOW:
        logger.info("Drift: insufficient rows (n=%d) for stable comparison", n)
        return None

    ref_df = df.iloc[:split]
    cur_df = df.iloc[split:]

    num_cols = _pick_numeric_columns(df, target_col)
    if not num_cols:
        return None

    ms_scores: list[float] = []
    ks_scores: list[float] = []

    for col in num_cols:
        r = ref_df[col].to_numpy(dtype=float)
        c = cur_df[col].to_numpy(dtype=float)
        ms, ks = _feature_drift_components(r, c)
        ms_scores.append(ms)
        if ks is not None:
            ks_scores.append(ks)

    mean_ms = float(np.mean(ms_scores)) if ms_scores else 0.0

    if ks_scores:
        mean_ks = float(np.mean(ks_scores))
        drift_score = 0.5 * mean_ms + 0.5 * mean_ks
        ks_used = True
    else:
        drift_score = mean_ms
        ks_used = False

    drift_score = float(min(max(drift_score, 0.0), 1.0))

    if drift_score < 1.0 / 3.0:
        status = "low"
    elif drift_score < 2.0 / 3.0:
        status = "medium"
    else:
        status = "high"

    return {
        "drift_score": drift_score,
        "status": status,
        "ks_used": ks_used,
        "n_features": len(num_cols),
    }
