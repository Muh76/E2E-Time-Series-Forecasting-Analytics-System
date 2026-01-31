"""
Inference script for trained forecasting models.

Responsibilities:
- Load trained model artifact
- Load recent historical data
- Run feature pipeline
- Generate horizon-step forecasts
- Output predictions in a clean DataFrame

Inference is fast, deterministic, and does not retrain. Run from project root:
    python scripts/inference.py
    python scripts/inference.py --model primary_lightgbm --horizon 14 --output predictions.parquet
"""

import argparse
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "primary_lightgbm"
MODEL_ARTIFACT_NAMES = {"primary_lightgbm", "baseline_seasonal_naive"}


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


def build_predict_config(config: dict) -> dict:
    """Build config for model.predict (no training params). Deterministic."""
    fe_cfg = config.get("feature_engineering") or {}
    model_cfg = config.get("model") or {}
    train_cfg = config.get("training") or {}
    freq = (model_cfg.get("params") or {}).get("frequency", "D")
    seed = int((train_cfg.get("runtime") or {}).get("seed", 42))
    return {
        "date_column": fe_cfg.get("date_column", "date"),
        "target_column": fe_cfg.get("target_column", "target_cleaned"),
        "entity_column": fe_cfg.get("entity_column"),
        "frequency": freq,
        "seed": seed,
        "feature_engineering": config.get("feature_engineering"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a trained forecasting model.")
    parser.add_argument(
        "--model",
        type=str,
        choices=sorted(MODEL_ARTIFACT_NAMES),
        default=DEFAULT_MODEL_NAME,
        help="Model artifact name (default: primary_lightgbm).",
    )
    parser.add_argument(
        "--processed-file",
        type=Path,
        default=None,
        help="Path to processed parquet (default: data/processed/etl_output.parquet from config).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Forecast horizon in steps (default: from config model.params.horizon_steps).",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Config env (e.g. local). Uses APP_ENV if not set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write predictions (CSV or parquet).",
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

    model_path = models_dir / f"{args.model}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}. Run scripts/train.py first.")
    if not processed_file.exists():
        raise FileNotFoundError(f"Processed data not found: {processed_file}")

    # Load model (no retraining)
    model = joblib.load(model_path)
    logger.info("Loaded model: %s", model_path)

    # Load recent historical data
    df = pd.read_parquet(processed_file)
    logger.info("Loaded historical data: path=%s, rows=%d", processed_file, len(df))

    # Run feature pipeline (same as training; deterministic)
    featured = run_feature_pipeline(df, config)
    logger.info("Feature pipeline done: rows=%d", len(featured))

    # Horizon
    model_cfg = config.get("model") or {}
    horizon = args.horizon or int((model_cfg.get("params") or {}).get("horizon_steps", 30))
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    # Predict (deterministic; no fit)
    predict_config = build_predict_config(config)
    predictions = model.predict(featured, horizon, predict_config)
    logger.info("Generated forecasts: horizon=%d, rows=%d", horizon, len(predictions))

    # Clean DataFrame: entity_id, date, y_pred, model_name (already from model.predict)
    out = predictions.copy()
    if "model_name" not in out.columns:
        out["model_name"] = args.model
    # Ensure column order
    cols = [c for c in ["entity_id", "date", "y_pred", "model_name"] if c in out.columns]
    out = out[cols]

    # Output
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".csv":
            out.to_csv(output_path, index=False)
        else:
            out.to_parquet(output_path, index=False)
        logger.info("Saved predictions: %s (%d rows)", output_path, len(out))
    else:
        # Print clean DataFrame to stdout
        display = out if len(out) <= 50 else out.head(50)
        if len(out) > 50:
            print(display.to_string() + f"\n... ({len(out)} rows total)")
        else:
            print(display.to_string())

    return out


if __name__ == "__main__":
    main()
