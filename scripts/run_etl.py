"""
Run the ETL pipeline end-to-end.

Loads config, executes ingest → validate → clean → augment, saves the
result as a Parquet file, and logs a summary.

Usage (from project root):
    python scripts/run_etl.py
    python scripts/run_etl.py --raw-file data/raw/target.csv --env local
    python scripts/run_etl.py --pipeline-mode rossmann
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

# Project root = parent of scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.etl.pipeline import run_pipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(env: str | None = None) -> dict:
    """Load base YAML config, optionally merging an environment overlay."""
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
                if (
                    key in config
                    and isinstance(config[key], dict)
                    and isinstance(val, dict)
                ):
                    config[key] = {**config[key], **val}
                else:
                    config[key] = val
    return config


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_paths(
    args: argparse.Namespace,
    data_cfg: dict,
) -> tuple[Path, Path]:
    """Determine absolute input and output paths from args + config."""
    raw_dir = PROJECT_ROOT / data_cfg.get("raw_path", "data/raw")
    processed_dir = PROJECT_ROOT / data_cfg.get("processed_path", "data/processed")

    if args.pipeline_mode == "rossmann":
        raw_file = raw_dir / "rossmann"
    else:
        raw_file = args.raw_file or raw_dir / "target.csv"

    if not raw_file.is_absolute():
        raw_file = PROJECT_ROOT / raw_file

    output_file = args.output_file or processed_dir / "etl_output.parquet"
    if not output_file.is_absolute():
        output_file = PROJECT_ROOT / output_file

    return raw_file, output_file


def _build_pipeline_config(
    raw_file: Path,
    config: dict,
    pipeline_mode: str,
) -> dict:
    """Assemble the sub-configs passed to ``run_pipeline``."""
    ingest_cfg: dict = {"path": str(raw_file), **config.get("ingest", {})}
    if pipeline_mode == "rossmann":
        ingest_cfg["rossmann_dir"] = str(raw_file)

    return {
        "ingest": ingest_cfg,
        "validate": config.get("validate", {}),
        "clean": config.get("clean", {}),
        "augment": config.get("augment", {}),
    }


# ---------------------------------------------------------------------------
# Summary logging
# ---------------------------------------------------------------------------

def _log_summary(df: pd.DataFrame, pipeline_mode: str, pipeline_config: dict) -> None:
    """Log a human-readable summary of the ETL output."""
    if pipeline_mode == "rossmann":
        _log_rossmann_summary(df)
    else:
        _log_generic_summary(df, pipeline_config)


def _log_rossmann_summary(df: pd.DataFrame) -> None:
    date_col = "date"
    store_col = "store_id"

    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    else:
        dates = pd.Series(dtype="datetime64[ns]")

    n_stores = df[store_col].nunique() if store_col in df.columns else 0

    if len(dates) > 0:
        logger.info(
            "Summary: dataset=rossmann, rows=%d, date_min=%s, date_max=%s, stores=%d",
            len(df), dates.min(), dates.max(), n_stores,
        )
    else:
        logger.info(
            "Summary: dataset=rossmann, rows=%d, stores=%d",
            len(df), n_stores,
        )


def _log_generic_summary(df: pd.DataFrame, pipeline_config: dict) -> None:
    date_col = (
        (pipeline_config.get("validate") or {}).get("date_column")
        or (pipeline_config.get("clean") or {}).get("date_column")
        or "date"
    )

    if date_col not in df.columns:
        logger.info("Summary: rows=%d", len(df))
        return

    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if len(dates) > 0:
        logger.info(
            "Summary: rows=%d, date_min=%s, date_max=%s",
            len(df), dates.min(), dates.max(),
        )
    else:
        logger.info("Summary: rows=%d", len(df))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ETL pipeline and save processed data.",
    )
    parser.add_argument(
        "--raw-file",
        type=Path,
        default=None,
        help="Path to raw CSV (default: data/raw/target.csv). "
             "Ignored when --pipeline-mode rossmann.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output parquet path (default: data/processed/etl_output.parquet).",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Config env to merge (e.g. local, staging, prod). "
             "Uses APP_ENV if not set.",
    )
    parser.add_argument(
        "--pipeline-mode",
        type=str,
        choices=("generic", "retail", "rossmann"),
        default="retail",
        help="Pipeline mode: retail, rossmann, or generic.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation step.",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable augmentation.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    env = args.env or os.environ.get("APP_ENV")
    config = load_config(env)
    data_cfg = config.get("data") or {}

    raw_file, output_file = _resolve_paths(args, data_cfg)
    pipeline_config = _build_pipeline_config(raw_file, config, args.pipeline_mode)

    logger.info(
        "Starting ETL: raw=%s, output=%s, mode=%s",
        raw_file, output_file, args.pipeline_mode,
    )

    df = run_pipeline(
        raw_file,
        config=pipeline_config,
        run_validation=not args.no_validate,
        run_augment=False if args.no_augment else None,
        pipeline_mode=args.pipeline_mode,
    )

    if df.empty:
        logger.warning("Pipeline produced no rows; not writing output.")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)
    logger.info("Saved processed data to %s", output_file)

    _log_summary(df, args.pipeline_mode, pipeline_config)


if __name__ == "__main__":
    main()
