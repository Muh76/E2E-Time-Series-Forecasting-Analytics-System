"""
Controlled synthetic data augmentation for daily time series.

Adds configurable noise or scaling for robustness (e.g. sensitivity analysis).
Deterministic when seed is fixed; no model logic. Use only when enabled in config.
"""

from typing import Any

import numpy as np
import pandas as pd


def add_gaussian_noise(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Add Gaussian noise to the value column. Deterministic when seed is set.

    Args:
        df: DataFrame with a numeric value column.
        config: Optional augment config. Expected keys (all optional):
            - value_column: Name of value column (default "target").
            - noise_std: Standard deviation of noise (default 0.0 → no noise).
            - noise_scale: If set, noise_std = noise_scale * std(value); ignored if noise_std set.
            - inplace: If True, modify DataFrame in place (default False).

        seed: Random seed for reproducibility. If None, behavior is non-deterministic.

    Returns:
        New DataFrame (or same if inplace=True) with value = value + noise.
    """
    cfg = config or {}
    value_col = cfg.get("value_column", "target")
    noise_std = cfg.get("noise_std")
    noise_scale = cfg.get("noise_scale")
    inplace = cfg.get("inplace", False)

    if value_col not in df.columns:
        return df if not inplace else df

    rng = np.random.default_rng(seed)

    if noise_std is None and noise_scale is not None:
        std_val = df[value_col].std()
        noise_std = float(noise_scale) * (std_val if pd.notna(std_val) and std_val != 0 else 1.0)
    if noise_std is None or noise_std <= 0:
        return df if not inplace else df

    out = df if inplace else df.copy()
    noise = rng.normal(0, noise_std, size=len(out))
    out[value_col] = out[value_col].astype(float) + noise
    return out


def augment(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Apply controlled augmentation if enabled in config.

    Only applies steps that are explicitly enabled (e.g. add_noise: true).
    Deterministic when seed is provided.

    Args:
        df: Cleaned DataFrame (daily time series).
        config: Optional augment config. Expected keys (all optional):
            - enabled: If False, return df unchanged (default False).
            - add_noise: If True, apply add_gaussian_noise with sub-keys
              (value_column, noise_std or noise_scale, inplace).
            - seed: Override seed for this run (otherwise use augment seed arg).

    Returns:
        DataFrame with augmentation applied, or unchanged if disabled.
    """
    cfg = config or {}
    if not cfg.get("enabled", False):
        return df

    use_seed = cfg.get("seed", seed)
    out = df

    if cfg.get("add_noise", False):
        noise_cfg = {k: v for k, v in cfg.items() if k not in ("enabled", "seed", "add_noise")}
        out = add_gaussian_noise(out, config=noise_cfg, seed=use_seed)

    return out


# ---------------------------------------------------------------------------
# Deterministic time series augmentation (missing blocks, noise shift, trend)
# ---------------------------------------------------------------------------

def augment_timeseries(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Apply deterministic synthetic augmentations to a time series DataFrame.

    Does not modify the original target column: adds target_original (copy of
    target) and target_augmented (with augmentations applied). Adds
    augmentation_type to mark which rows were modified and how.

    Augmentations (each config-driven and reproducible via seed):
    - Missing data blocks: set contiguous date blocks to NaN in augmented series.
    - Noise regime shifts: add Gaussian noise with different scale before/after
      random cut point(s).
    - Trend change: add a linear trend over random date window(s).

    Args:
        df: DataFrame with date column and value column; optional entity column.
        config: Optional. Keys (all optional):
            - value_column: default "target".
            - date_column: default "date".
            - entity_column: if set, augmentations are applied per entity (e.g. store_id).
            - output_original_column: default "target_original".
            - output_augmented_column: default "target_augmented".
            - output_type_column: default "augmentation_type".
            - missing_blocks: dict with enabled, n_blocks, block_size_min, block_size_max.
            - noise_regime_shift: dict with enabled, n_shifts, scale_before, scale_after.
            - trend_change: dict with enabled, n_windows, window_length_min, window_length_max,
              slope_min, slope_max.
        seed: Random seed for reproducibility. Required for deterministic behavior.

    Returns:
        New DataFrame with original column preserved, augmented value column,
        and augmentation_type column. Sorted by date (and entity if present).
    """
    cfg = config or {}
    date_col = cfg.get("date_column", "date")
    value_col = cfg.get("value_column", "target")
    entity_col = cfg.get("entity_column")
    out_orig = cfg.get("output_original_column", "target_original")
    out_aug = cfg.get("output_augmented_column", "target_augmented")
    out_type = cfg.get("output_type_column", "augmentation_type")

    for c in (date_col, value_col):
        if c not in df.columns:
            raise ValueError(f"augment_timeseries requires column '{c}'. Found: {list(df.columns)}.")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
    out = out.sort_values([c for c in [date_col, entity_col] if c in out.columns]).reset_index(drop=True)
    out[out_orig] = out[value_col].astype(float)
    out[out_aug] = out[value_col].astype(float).copy()
    out[out_type] = "original"

    rng = np.random.default_rng(seed)

    if entity_col and entity_col in out.columns:
        groups = list(out.groupby(entity_col, sort=False))
    else:
        groups = [(None, out)]

    for entity_id, grp in groups:
        if entity_id is not None:
            mask = out[entity_col] == entity_id
            idx = out.index[mask]
        else:
            idx = out.index
        n = len(idx)
        if n == 0:
            continue

        # 1. Missing data blocks
        mb = (cfg.get("missing_blocks") or {}).get("enabled", False)
        if mb and n > 0:
            n_blocks = (cfg.get("missing_blocks") or {}).get("n_blocks", 1)
            bs_min = (cfg.get("missing_blocks") or {}).get("block_size_min", 1)
            bs_max = (cfg.get("missing_blocks") or {}).get("block_size_max", 3)
            n_blocks = min(n_blocks, max(1, n // max(1, bs_min)))
            for _ in range(n_blocks):
                block_size = int(rng.integers(bs_min, bs_max + 1))
                start = int(rng.integers(0, max(1, n - block_size + 1)))
                for i in range(start, min(start + block_size, n)):
                    pos = idx[i]
                    out.loc[pos, out_aug] = np.nan
                    prev = out.loc[pos, out_type]
                    out.loc[pos, out_type] = "missing_block" if prev == "original" else f"{prev},missing_block"

        # 2. Noise regime shifts (segments between cut points get alternating scale)
        ns = (cfg.get("noise_regime_shift") or {}).get("enabled", False)
        if ns and n > 0:
            n_shifts = (cfg.get("noise_regime_shift") or {}).get("n_shifts", 1)
            scale_before = (cfg.get("noise_regime_shift") or {}).get("scale_before", 0.0)
            scale_after = (cfg.get("noise_regime_shift") or {}).get("scale_after", 1.0)
            cuts = sorted(set([0, n] + rng.integers(1, n, size=min(n_shifts, n - 1)).tolist()))
            for seg_idx in range(len(cuts) - 1):
                lo, hi = cuts[seg_idx], cuts[seg_idx + 1]
                scale = scale_before if seg_idx % 2 == 0 else scale_after
                noise = rng.normal(0, scale, size=hi - lo)
                out.loc[idx[lo:hi], out_aug] = out.loc[idx[lo:hi], out_aug].values + noise
                for i in range(lo, hi):
                    pos = idx[i]
                    prev = out.loc[pos, out_type]
                    out.loc[pos, out_type] = "noise_shift" if prev == "original" else f"{prev},noise_shift"

        # 3. Trend change over window
        tc = (cfg.get("trend_change") or {}).get("enabled", False)
        if tc and n > 0:
            n_windows = (cfg.get("trend_change") or {}).get("n_windows", 1)
            w_min = (cfg.get("trend_change") or {}).get("window_length_min", 5)
            w_max = (cfg.get("trend_change") or {}).get("window_length_max", 14)
            slope_min = (cfg.get("trend_change") or {}).get("slope_min", -1.0)
            slope_max = (cfg.get("trend_change") or {}).get("slope_max", 1.0)
            for _ in range(n_windows):
                wlen = int(rng.integers(w_min, w_max + 1))
                wlen = min(wlen, n)
                start = int(rng.integers(0, max(1, n - wlen + 1)))
                slope = float(rng.uniform(slope_min, slope_max))
                dates = out.loc[idx, date_col].iloc[start : start + wlen]
                t0 = dates.iloc[0]
                for j in range(wlen):
                    i = start + j
                    pos = idx[i]
                    dt = (dates.iloc[j] - t0).days
                    out.loc[pos, out_aug] = out.loc[pos, out_aug] + slope * dt
                    prev = out.loc[pos, out_type]
                    out.loc[pos, out_type] = "trend" if prev == "original" else f"{prev},trend"

    return out
