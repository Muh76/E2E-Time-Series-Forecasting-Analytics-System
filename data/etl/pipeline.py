"""
ETL pipeline orchestrator for daily time series.

Runs ingest → validate → clean → (optional) augment in order.
Config-driven; each step receives its slice of config. Deterministic
when augment is disabled or seeded. No model logic.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from . import augment, clean, ingest, validate


def run_pipeline(
    raw_path: str | Path,
    config: dict[str, Any] | None = None,
    *,
    output_path: str | Path | None = None,
    run_validation: bool = True,
    run_augment: bool | None = None,
) -> pd.DataFrame:
    """
    Run the full ETL pipeline: ingest, validate, clean, optionally augment.

    Steps:
    1. Ingest: load raw CSV from raw_path (config["ingest"]).
    2. Validate: schema and data checks (config["validate"]). If run_validation
       is True and validation fails, raises ValueError with error messages.
    3. Clean: normalize dates and clean values (config["clean"]).
    4. Augment: optional synthetic augmentation (config["augment"]) only if
       run_augment is True or config["augment"]["enabled"] is True.

    Args:
        raw_path: Path to raw CSV file.
        config: Full pipeline config. Expected top-level keys: ingest, validate,
            clean, augment. Each step receives its key (e.g. config["validate"]).
            If a key is missing, that step gets None.
        output_path: If set, write the final DataFrame to this path (CSV or
            parquet by suffix). Not implemented here — placeholder for caller
            or loader module.
        run_validation: If True, fail on validation errors (default True).
        run_augment: If True, run augment step. If False, skip. If None, use
            config["augment"]["enabled"].

    Returns:
        Final DataFrame after all steps.

    Raises:
        FileNotFoundError: If raw_path does not exist.
        ValueError: If run_validation is True and validation fails.
    """
    cfg = config or {}
    ingest_cfg = cfg.get("ingest")
    validate_cfg = cfg.get("validate")
    clean_cfg = cfg.get("clean")
    augment_cfg = cfg.get("augment")

    # 1. Ingest
    df = ingest.load_raw_csv(raw_path, config=ingest_cfg)
    if df.empty:
        if output_path:
            _write_placeholder(output_path, df)
        return df

    # 2. Validate
    if run_validation:
        schema_result = validate.validate_schema(df, validate_cfg)
        if not schema_result.valid:
            raise ValueError(f"Schema validation failed: {schema_result.errors}")
        data_result = validate.validate_data(df, validate_cfg)
        if not data_result.valid:
            raise ValueError(f"Data validation failed: {data_result.errors}")

    # 3. Clean
    df = clean.clean(df, clean_cfg)

    # 4. Augment (optional)
    do_augment = run_augment if run_augment is not None else (augment_cfg or {}).get("enabled", False)
    if do_augment and augment_cfg:
        df = augment.augment(df, config=augment_cfg, seed=augment_cfg.get("seed"))

    if output_path is not None:
        _write_placeholder(output_path, df)

    return df


def _write_placeholder(output_path: str | Path, df: pd.DataFrame) -> None:
    """
    Placeholder: write DataFrame to output_path based on suffix.

    Callers may replace this with a proper loader (e.g. to_parquet, GCS).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
