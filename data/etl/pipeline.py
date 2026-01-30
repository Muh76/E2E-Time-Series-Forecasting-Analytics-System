"""
ETL pipeline orchestrator for daily time series.

Runs ingest → validate → clean → (optional) augment in order.
Config-driven; each step receives its slice of config. Logs each step.
No file writing in this module. Deterministic when augment is disabled or seeded.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from . import augment, clean, ingest, validate

logger = logging.getLogger(__name__)


def run_pipeline(
    raw_path: str | Path,
    config: dict[str, Any] | None = None,
    *,
    run_validation: bool = True,
    run_augment: bool | None = None,
    pipeline_mode: str = "generic",
) -> pd.DataFrame:
    """
    Run the full ETL pipeline: ingest, validate, clean, optionally augment.

    Calls ingest, validate, clean, and augment in order. Accepts a config
    object; logs each step. Returns the final processed DataFrame. No file
    writing is performed in this module.

    Steps:
    1. Ingest: load raw CSV (generic or retail per pipeline_mode).
    2. Validate: schema and data checks. If run_validation is True and
       validation fails, raises ValueError.
    3. Clean: normalize dates and clean values (generic or retail).
    4. Augment: optional augmentation if run_augment or config enables it.

    Args:
        raw_path: Path to raw CSV file (or use config["ingest"]["path"] when
            pipeline_mode is "retail" and path is passed in config).
        config: Full pipeline config. Top-level keys: ingest, validate, clean,
            augment. Each step receives its slice (e.g. config["validate"]).
        run_validation: If True, fail on validation errors (default True).
        run_augment: If True, run augment step. If False, skip. If None, use
            config["augment"]["enabled"] or config["augment_timeseries"] enable.
        pipeline_mode: "generic" (default) or "retail". Retail uses
            load_retail_sales_csv, validate_retail, clean_retail, and
            optionally augment_timeseries.

    Returns:
        Final processed DataFrame after all steps.

    Raises:
        FileNotFoundError: If raw_path (or config path) does not exist.
        ValueError: If run_validation is True and validation fails.
    """
    cfg = config or {}
    ingest_cfg = cfg.get("ingest")
    validate_cfg = cfg.get("validate")
    clean_cfg = cfg.get("clean")
    augment_cfg = cfg.get("augment")

    # 1. Ingest
    path = (ingest_cfg or {}).get("path") or raw_path
    logger.info("ETL step: ingest (path=%s)", path)
    if pipeline_mode == "retail":
        df = ingest.load_retail_sales_csv(path=path, config=ingest_cfg)
    else:
        df = ingest.load_raw_csv(path, config=ingest_cfg)
    logger.info("ETL step: ingest done (rows=%d)", len(df))
    if df.empty:
        return df

    # 2. Validate
    logger.info("ETL step: validate")
    if run_validation:
        if pipeline_mode == "retail":
            validate.validate_retail(df, config=validate_cfg)
        else:
            schema_result = validate.validate_schema(df, validate_cfg)
            if not schema_result.valid:
                raise ValueError(f"Schema validation failed: {schema_result.errors}")
            data_result = validate.validate_data(df, validate_cfg)
            if not data_result.valid:
                raise ValueError(f"Data validation failed: {data_result.errors}")
    logger.info("ETL step: validate done")

    # 3. Clean
    logger.info("ETL step: clean")
    if pipeline_mode == "retail":
        df = clean.clean_retail(df, config=clean_cfg)
    else:
        df = clean.clean(df, config=clean_cfg)
    logger.info("ETL step: clean done (rows=%d)", len(df))

    # 4. Augment (optional)
    use_timeseries_augment = (augment_cfg or {}).get("timeseries_enabled", False)
    do_augment = run_augment if run_augment is not None else (
        (augment_cfg or {}).get("enabled", False) or use_timeseries_augment
    )
    if do_augment and augment_cfg:
        logger.info("ETL step: augment")
        if use_timeseries_augment:
            df = augment.augment_timeseries(df, config=augment_cfg, seed=augment_cfg.get("seed"))
        else:
            df = augment.augment(df, config=augment_cfg, seed=augment_cfg.get("seed"))
        logger.info("ETL step: augment done (rows=%d)", len(df))

    logger.info("ETL pipeline complete")
    return df
