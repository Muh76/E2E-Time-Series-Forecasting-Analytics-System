"""
RAG service: unified FAISS index and chunk metadata loading.

- FAISS index: data/faiss_index.bin
- Metadata: data/chunk_metadata.pkl
No data/indices/* paths. Ingestion logic is separate.
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Unified paths (no data/indices/*)
FAISS_INDEX_PATH = "data/faiss_index.bin"
CHUNK_METADATA_PATH = "data/chunk_metadata.pkl"

_index = None
_metadata = None


def _resolve_path(relative_path: str, base_dir: Path | None = None) -> Path:
    """Resolve path relative to project root or given base."""
    path = Path(relative_path)
    if path.is_absolute():
        return path
    base = base_dir or Path.cwd()
    return (base / path).resolve()


def load_faiss_index(index_path: str | Path | None = None, base_dir: Path | None = None) -> Any:
    """
    Load FAISS index from data/faiss_index.bin (or given path).
    Logs index ntotal. Does not use data/indices/*.
    """
    global _index
    path = _resolve_path(index_path or FAISS_INDEX_PATH, base_dir)
    if not path.exists():
        logger.warning("FAISS index not found at %s", path)
        return None
    try:
        import faiss
        _index = faiss.read_index(str(path))
        ntotal = _index.ntotal
        logger.info("FAISS index loaded: path=%s, ntotal=%d", path, ntotal)
        return _index
    except Exception as e:
        logger.exception("Failed to load FAISS index from %s: %s", path, e)
        return None


def load_chunk_metadata(metadata_path: str | Path | None = None, base_dir: Path | None = None) -> list[Any] | None:
    """
    Load chunk metadata from data/chunk_metadata.pkl (or given path).
    Logs number of chunks. Does not use data/indices/*.
    """
    global _metadata
    path = _resolve_path(metadata_path or CHUNK_METADATA_PATH, base_dir)
    if not path.exists():
        logger.warning("Chunk metadata not found at %s", path)
        return None
    try:
        import pickle
        with open(path, "rb") as f:
            _metadata = pickle.load(f)
        # Support list or dict-like (e.g. list of dicts)
        num_chunks = len(_metadata) if _metadata is not None else 0
        logger.info("Chunk metadata loaded: path=%s, num_chunks=%d", path, num_chunks)
        return _metadata
    except Exception as e:
        logger.exception("Failed to load chunk metadata from %s: %s", path, e)
        return None


def get_faiss_and_metadata(
    index_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    base_dir: Path | None = None,
    reload: bool = False,
) -> tuple[Any | None, list[Any] | None]:
    """
    Load FAISS index and chunk metadata from unified paths.
    Returns (index, metadata). Logs FAISS ntotal and number of chunks.
    """
    global _index, _metadata
    if reload:
        _index = None
        _metadata = None
    if _index is None:
        load_faiss_index(index_path, base_dir)
    if _metadata is None:
        load_chunk_metadata(metadata_path, base_dir)
    if _index is not None and _metadata is not None:
        logger.info("RAG loaded: FAISS ntotal=%d, number of chunks loaded=%d", _index.ntotal, len(_metadata))
    return _index, _metadata


def get_index() -> Any | None:
    """Return currently loaded FAISS index (or None)."""
    return _index


def get_metadata() -> list[Any] | None:
    """Return currently loaded chunk metadata (or None)."""
    return _metadata
