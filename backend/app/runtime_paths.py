"""
Central path resolution for the backend.

Set ``E2E_PROJECT_ROOT`` in production so data, config, and artifacts are not
tied to the install location. Optional overrides target individual assets.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def project_root() -> Path:
    """Repository / deployment root (directory that contains ``data/``, ``config/``)."""
    root = os.environ.get("E2E_PROJECT_ROOT", "").strip()
    if root:
        return Path(root).expanduser().resolve()
    # backend/app/runtime_paths.py -> parents[2] == project root
    return Path(__file__).resolve().parent.parent.parent


def ensure_project_on_sys_path() -> Path:
    """Ensure project root is importable (``data.*``, ``models.*``)."""
    root = project_root()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


def processed_parquet_path() -> Path:
    """Processed features parquet (default ``data/processed/etl_output.parquet``)."""
    p = os.environ.get("E2E_PROCESSED_PARQUET_PATH", "").strip()
    if p:
        return Path(p).expanduser().resolve()
    return project_root() / "data" / "processed" / "etl_output.parquet"


def base_default_config_path() -> Path:
    """Base training / feature config YAML."""
    p = os.environ.get("E2E_BASE_CONFIG_PATH", "").strip()
    if p:
        return Path(p).expanduser().resolve()
    return project_root() / "config" / "base" / "default.yaml"


def env_config_path() -> Path:
    """Environment overlay: ``config/{APP_ENV}/config.yaml``."""
    env = os.environ.get("APP_ENV", "local")
    return project_root() / "config" / env / "config.yaml"


def artifacts_models_dir() -> Path:
    """Directory containing ``primary_lightgbm.joblib`` and related files."""
    p = os.environ.get("E2E_MODEL_ARTIFACTS_DIR", "").strip()
    if p:
        return Path(p).expanduser().resolve()
    return project_root() / "artifacts" / "models"


def resolve_under_root(relative: str | Path) -> Path:
    """Resolve ``relative`` against project root unless already absolute."""
    path = Path(relative)
    if path.is_absolute():
        return path.resolve()
    return (project_root() / path).resolve()


def faiss_index_file() -> Path:
    """FAISS index file for RAG (override with ``E2E_FAISS_INDEX_PATH``)."""
    rel = os.environ.get("E2E_FAISS_INDEX_PATH", "data/faiss_index.bin").strip()
    return resolve_under_root(rel)


def chunk_metadata_file() -> Path:
    """Chunk metadata pickle for RAG (override with ``E2E_CHUNK_METADATA_PATH``)."""
    rel = os.environ.get("E2E_CHUNK_METADATA_PATH", "data/chunk_metadata.pkl").strip()
    return resolve_under_root(rel)
