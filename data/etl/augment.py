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
            - value_column: Name of value column (default "value").
            - noise_std: Standard deviation of noise (default 0.0 → no noise).
            - noise_scale: If set, noise_std = noise_scale * std(value); ignored if noise_std set.
            - inplace: If True, modify DataFrame in place (default False).

        seed: Random seed for reproducibility. If None, behavior is non-deterministic.

    Returns:
        New DataFrame (or same if inplace=True) with value = value + noise.
    """
    cfg = config or {}
    value_col = cfg.get("value_column", "value")
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
