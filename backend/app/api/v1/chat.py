"""
Chat / RAG API routes. Use unified FAISS and metadata paths only.
No fallback to data/indices/*.
"""

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from app.services.rag_service import (
    FAISS_INDEX_PATH,
    CHUNK_METADATA_PATH,
    get_faiss_and_metadata,
    get_index,
    get_metadata,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


def _ensure_loaded(base_dir: Path | None = None) -> tuple[Any, Any]:
    """
    Ensure FAISS index and metadata are loaded from unified paths.
    Uses data/faiss_index.bin and data/chunk_metadata.pkl only (no data/indices/*).
    """
    index = get_index()
    metadata = get_metadata()
    if index is None or metadata is None:
        index, metadata = get_faiss_and_metadata(
            index_path=FAISS_INDEX_PATH,
            metadata_path=CHUNK_METADATA_PATH,
            base_dir=base_dir,
            reload=False,
        )
    if index is None:
        logger.warning("Chat: FAISS index not available (path=%s)", FAISS_INDEX_PATH)
        raise HTTPException(503, "RAG index not available")
    if metadata is None:
        logger.warning("Chat: chunk metadata not available (path=%s)", CHUNK_METADATA_PATH)
        raise HTTPException(503, "RAG metadata not available")
    return index, metadata


@router.post("/query")
async def chat_query(body: dict[str, str] | None = None):
    """
    RAG chat query. Loads index and metadata from data/faiss_index.bin and
    data/chunk_metadata.pkl only. No fallback to data/indices/*.
    Logs FAISS ntotal and number of chunks.
    """
    message = (body or {}).get("message", "")
    index, metadata = _ensure_loaded()
    ntotal = index.ntotal
    num_chunks = len(metadata) if metadata else 0
    logger.info("Chat query: FAISS ntotal=%d, num_chunks=%d", ntotal, num_chunks)
    # Placeholder: actual search would use index.search(...) and metadata for context
    return {
        "message": message,
        "ntotal": ntotal,
        "num_chunks": num_chunks,
        "status": "ok",
    }
