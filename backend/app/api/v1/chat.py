"""
Chat / RAG API routes. Use unified FAISS and metadata paths only.
No fallback to data/indices/*.
"""

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.app.runtime_paths import chunk_metadata_file, faiss_index_file
from backend.app.services.rag_service import get_faiss_and_metadata, get_index, get_metadata

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


def _ensure_loaded(base_dir: Path | None = None) -> tuple[Any, Any]:
    """
    Ensure FAISS index and metadata are loaded from project-relative paths
    (see ``E2E_FAISS_INDEX_PATH`` / ``E2E_CHUNK_METADATA_PATH``).
    """
    index = get_index()
    metadata = get_metadata()
    idx_path = faiss_index_file()
    meta_path = chunk_metadata_file()
    if index is None or metadata is None:
        index, metadata = get_faiss_and_metadata(
            index_path=idx_path,
            metadata_path=meta_path,
            base_dir=base_dir,
            reload=False,
        )
    if index is None:
        logger.warning("Chat: FAISS index not available (path=%s)", idx_path)
        raise HTTPException(503, "RAG index not available")
    if metadata is None:
        logger.warning("Chat: chunk metadata not available (path=%s)", meta_path)
        raise HTTPException(503, "RAG metadata not available")
    return index, metadata


@router.post("/query")
async def chat_query(body: dict[str, str] | None = None):
    """
    Accept a chat message and report index health. Vector search over the index
    is not enabled in this build (no query embedding pipeline); use Insight
    Copilot for rule-based explanations.
    """
    message = (body or {}).get("message", "").strip()
    index, metadata = _ensure_loaded()
    ntotal = index.ntotal
    num_chunks = len(metadata) if metadata else 0
    logger.info(
        "chat_query: message_len=%d faiss_ntotal=%d metadata_chunks=%d",
        len(message),
        ntotal,
        num_chunks,
    )
    return {
        "status": "index_ready",
        "message_received": message,
        "index": {"vector_count": ntotal, "chunk_count": num_chunks},
        "reply": (
            "RAG vector search is not configured for this deployment (no embedding step). "
            "The FAISS index and chunk metadata are loaded; use POST /api/v1/copilot or "
            "/api/v1/copilot/explain for grounded text responses."
        ),
    }
