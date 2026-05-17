"""
RAG service: FAISS index loading and vector search.

- FAISS index: data/faiss_index.bin   (built by scripts/generate_rag_index.py)
- Metadata:    data/chunk_metadata.pkl
No data/indices/* paths. Ingestion logic is separate.
"""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

# HuggingFace fast tokenizers use Rust-based parallelism that deadlocks in
# forked processes (e.g. uvicorn workers on macOS).  Must be set before any
# tokenizers import; setting it here ensures it's always in effect.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from backend.app.runtime_paths import project_root  # noqa: E402

logger = logging.getLogger(__name__)

# Relative defaults under project root (override with E2E_* env; see runtime_paths)
FAISS_INDEX_PATH = "data/faiss_index.bin"
CHUNK_METADATA_PATH = "data/chunk_metadata.pkl"

_index = None
_metadata = None
_embed_model = None


def _resolve_path(relative_path: str, base_dir: Path | None = None) -> Path:
    """Resolve path relative to project root (or ``base_dir`` when provided)."""
    path = Path(relative_path)
    if path.is_absolute():
        return path.resolve()
    base = base_dir or project_root()
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


# ---------------------------------------------------------------------------
# Embedding model (singleton, lazy-loaded on first search)
# ---------------------------------------------------------------------------


def _get_embed_model() -> Any | None:
    """Lazy-load the sentence-transformers model once; return None on import failure."""
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    try:
        from sentence_transformers import SentenceTransformer

        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("RAG embedding model loaded: all-MiniLM-L6-v2")
    except ImportError:
        logger.error("sentence-transformers not installed; run: pip install sentence-transformers")
    except Exception as exc:
        logger.exception("Failed to load embedding model: %s", exc)
    return _embed_model


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------


def search(query: str, k: int = 5) -> list[dict]:
    """
    Embed *query* and return the top-k most similar chunks from the FAISS index.

    Each result is a dict with keys: source, header, text, score.
    Returns [] when the index is not loaded, the embedding model is unavailable,
    or an unexpected error occurs — callers should degrade gracefully.
    """
    if _index is None or _metadata is None:
        logger.warning("RAG search: index or metadata not loaded")
        return []

    model = _get_embed_model()
    if model is None:
        return []

    try:
        vec = model.encode([query], normalize_embeddings=True)
        vec = np.array(vec, dtype="float32")
        n = min(k, _index.ntotal)
        scores, indices = _index.search(vec, n)
        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(_metadata):
                chunk = dict(_metadata[idx])
                chunk["score"] = float(score)
                results.append(chunk)
        return results
    except Exception as exc:
        logger.exception("RAG search failed: %s", exc)
        return []
