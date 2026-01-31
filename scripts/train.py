"""
Training script for forecasting models.

Responsibilities:
- Load config
- Load processed data
- Run feature pipeline
- Time-based train/validation split
- Train baseline (seasonal naive) and primary (LightGBM) model
- Evaluate both models (MAE, RMSE, MAPE)
- Save trained model artifacts to artifacts/models/
- Log metrics clearly

No FastAPI, no inference serving. Run from project root:
    python scripts/train.py
    python scripts/train.py --env local --processed-file data/processed/etl_output.parquet
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Project root = parent of scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import yaml

from data.feature_engineering import run_feature_pipeline
from models.evaluation import compute_metrics
from models.forecasting import LightGBMForecast, SeasonalNaiveForecast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(env: str | None = None) -> dict:
    """Load base config and optionally merge with env-specific config."""
    config_dir = PROJECT_ROOT / "config"
    base_path = config_dir / "base" / "default.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")

    with open(base_path) as f:
        config = yaml.safe_load(f) or {}

    if env:
        env_path = config_dir / env / "config.yaml"
        if env_path.exists():
            with open(env_path) as f:
                env_config = yaml.safe_load(f) or {}
            for key, val in env_config.items():
                if key in config and isinstance(config[key], dict) and isinstance(val, dict):
                    config[key] = {**config[key], **val}
                else:
                    config[key] = val
    return config


def time_based_split(
    df: pd.DataFrame,
    date_col: str,
    val_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time: train = rows before cutoff date, val = rows at or after cutoff."""
    dates = df[date_col].drop_duplicates().sort_values()
    n = len(dates)
    if n < 2 or val_frac <= 0 or val_frac >= 1:
        return df, df.iloc[0:0]
    cutoff_idx = max(0, int(n * (1 - val_frac)))
    cutoff_date = dates.iloc[cutoff_idx]
    train = df[df[date_col] < cutoff_date]
    val = df[df[date_col] >= cutoff_date]
    return train, val


def align_forecasts_to_actuals(
    forecasts: pd.DataFrame,
    actuals: pd.DataFrame,
    date_col: str,
    target_col: str,
    entity_col: str | None,
) -> tuple[list[float], list[float], list[object]]:
    """Merge forecast (entity_id, date, y_pred) with actuals (entity, date, target). Return y_true, y_pred, entity_ids."""
    if entity_col and "entity_id" in forecasts.columns and entity_col in actuals.columns:
        merged = forecasts.merge(
            actuals[[entity_col, date_col, target_col]],
            left_on=["entity_id", "date"],
            right_on=[entity_col, date_col],
            how="inner",
        )
        entity_ids = merged["entity_id"].tolist()
    else:
        merged = forecasts.merge(
            actuals[[date_col, target_col]],
            on=date_col,
            how="inner",
        )
        entity_ids = merged["entity_id"].tolist() if "entity_id" in merged.columns else [None] * len(merged)
    y_true = merged[target_col].tolist()
    y_pred = merged["y_pred"].tolist()
    return y_true, y_pred, entity_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Train forecasting models and save artifacts.")
    parser.add_argument(
        "--processed-file",
        type=Path,
        default=None,
        help="Path to processed parquet (default: data/processed/etl_output.parquet from config).",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Config env to merge (e.g. local, staging, prod). Uses APP_ENV if not set.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=None,
        help="Fraction of dates for validation (default: from config training.validation_split).",
    )
    args = parser.parse_args()

    env = args.env or os.environ.get("APP_ENV")
    config = load_config(env)
    logger.info("Config loaded (env=%s)", env)

    # Paths
    data_cfg = config.get("data") or {}
    processed_path = data_cfg.get("processed_path", "data/processed")
    artifacts_path = Path(data_cfg.get("artifacts_path", "artifacts"))
    models_dir = artifacts_path / "models"
    models_dir = PROJECT_ROOT / models_dir if not models_dir.is_absolute() else models_dir
    processed_file = args.processed_file or (PROJECT_ROOT / processed_path / "etl_output.parquet")
    if not processed_file.is_absolute():
        processed_file = PROJECT_ROOT / processed_file

    if not processed_file.exists():
        raise FileNotFoundError(f"Processed data not found: {processed_file}")

    # Load processed data
    df = pd.read_parquet(processed_file)
    logger.info("Loaded processed data: path=%s, rows=%d", processed_file, len(df))

    # Feature pipeline
    featured = run_feature_pipeline(df, config)
    logger.info("Feature pipeline done: rows=%d, cols=%d", len(featured), featured.shape[1])

    # Time-based split
    fe_cfg = config.get("feature_engineering") or {}
    date_col = fe_cfg.get("date_column", "date")
    target_col = fe_cfg.get("target_column", "target_cleaned")
    entity_col = fe_cfg.get("entity_column")

    train_cfg = config.get("training") or {}
    val_frac = args.validation_split
    if val_frac is None:
        val_frac = float((train_cfg.get("data") or {}).get("validation_split", 0.2))

    train_df, val_df = time_based_split(featured, date_col, val_frac)
    logger.info(
        "Time-based split: val_frac=%.2f, train_rows=%d, val_rows=%d",
        val_frac, len(train_df), len(val_df),
    )
    if val_df.empty:
        logger.warning("Validation set is empty; cannot evaluate. Train only.")

    # Model config (shared)
    model_cfg = config.get("model") or {}
    freq = (model_cfg.get("params") or {}).get("frequency", "D")
    seed = int((train_cfg.get("runtime") or {}).get("seed", 42))
    train_config = {
        "date_column": date_col,
        "target_column": target_col,
        "entity_column": entity_col,
        "frequency": freq,
        "seed": seed,
        "feature_engineering": config.get("feature_engineering"),
        "time_split_val_frac": 0.0,  # we already split; use full train_df for fitting
    }

    # Train baseline (seasonal naive)
    baseline = SeasonalNaiveForecast()
    baseline.fit(train_df, train_config)
    logger.info("Baseline (seasonal_naive) fitted")

    # Train primary (LightGBM)
    primary = LightGBMForecast()
    primary.fit(train_df, train_config)
    logger.info("Primary (LightGBM) fitted")

    # Evaluate on validation
    metrics_log: dict[str, dict[str, float]] = {}
    if not val_df.empty:
        val_dates = val_df[date_col].drop_duplicates().sort_values()
        horizon = len(val_dates)

        pred_baseline = baseline.predict(train_df, horizon, train_config)
        pred_primary = primary.predict(train_df, horizon, train_config)

        y_true_b, y_pred_b, eids_b = align_forecasts_to_actuals(
            pred_baseline, val_df, date_col, target_col, entity_col
        )
        y_true_p, y_pred_p, eids_p = align_forecasts_to_actuals(
            pred_primary, val_df, date_col, target_col, entity_col
        )

        metrics_baseline = compute_metrics(y_true_b, y_pred_b, entity_ids=eids_b if eids_b and eids_b[0] is not None else None)
        metrics_primary = compute_metrics(y_true_p, y_pred_p, entity_ids=eids_p if eids_p and eids_p[0] is not None else None)
        metrics_log["baseline_seasonal_naive"] = metrics_baseline
        metrics_log["primary_lightgbm"] = metrics_primary

        logger.info(
            "Validation metrics (baseline): MAE=%.4f, RMSE=%.4f, MAPE=%.2f%%",
            metrics_baseline["mae"], metrics_baseline["rmse"], metrics_baseline["mape"],
        )
        logger.info(
            "Validation metrics (primary):  MAE=%.4f, RMSE=%.4f, MAPE=%.2f%%",
            metrics_primary["mae"], metrics_primary["rmse"], metrics_primary["mape"],
        )

    # Save artifacts to artifacts/models/
    models_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = models_dir / "baseline_seasonal_naive.joblib"
    joblib.dump(baseline, baseline_path)
    logger.info("Saved baseline model: %s", baseline_path)

    primary_path = models_dir / "primary_lightgbm.joblib"
    joblib.dump(primary, primary_path)
    logger.info("Saved primary model: %s", primary_path)

    if metrics_log:
        metrics_path = models_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_log, f, indent=2)
        logger.info("Saved metrics: %s", metrics_path)

    logger.info("Training complete. Artifacts in %s", models_dir)


if __name__ == "__main__":
    main()
