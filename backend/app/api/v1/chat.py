"""
Chat / RAG API routes.

POST /api/v1/chat/query
  1. Validates that the FAISS index and chunk metadata are loaded.
  2. Embeds the user message and retrieves the top-5 most relevant documentation chunks.
  3. If OPENAI_API_KEY is set, calls OpenAI to produce a grounded natural-language reply.
     Otherwise, returns the top retrieved chunk as plain text.
  4. Returns: reply, sources, generated_at.

The index is built offline by scripts/generate_rag_index.py and loaded at API startup.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.runtime_paths import chunk_metadata_file, faiss_index_file
from backend.app.services.rag_service import get_faiss_and_metadata, get_index, get_metadata, search


class ChatRequest(BaseModel):
    message: str


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

_TOP_K = 5
_MAX_CONTEXT_CHARS = 6_000


def _ensure_loaded(base_dir: Path | None = None) -> tuple[Any, Any]:
    """Return loaded (index, metadata), triggering a lazy load if needed."""
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
        logger.warning("chat: FAISS index not available (path=%s)", idx_path)
        raise HTTPException(
            503,
            detail=(
                "RAG index not found. Run `python scripts/generate_rag_index.py` "
                "to build data/faiss_index.bin, then restart the API."
            ),
        )
    if metadata is None:
        logger.warning("chat: chunk metadata not available (path=%s)", meta_path)
        raise HTTPException(
            503,
            detail=(
                "RAG metadata not found. Run `python scripts/generate_rag_index.py` "
                "to build data/chunk_metadata.pkl, then restart the API."
            ),
        )
    return index, metadata


# ---------------------------------------------------------------------------
# OpenAI reply generation
# ---------------------------------------------------------------------------


async def _openai_reply(query: str, chunks: list[dict]) -> str | None:
    """
    Call OpenAI Chat Completions with retrieved context.
    Returns the reply string, or None if the key is absent or the call fails.
    """
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return None

    # Build context from top chunks, capped to avoid token bloat
    parts: list[str] = []
    total = 0
    for chunk in chunks:
        entry = f"[{chunk['source']}] {chunk['header']}\n{chunk['text']}"
        if total + len(entry) > _MAX_CONTEXT_CHARS:
            break
        parts.append(entry)
        total += len(entry)

    context = "\n\n---\n\n".join(parts) if parts else "(no context retrieved)"

    system_msg = (
        "You are a documentation assistant for an end-to-end time series forecasting system. "
        "Answer the user's question using only the retrieved documentation context provided. "
        "Cite the source file when it helps. Be concise and accurate. "
        "If the context does not cover the question, say so clearly instead of guessing."
    )
    user_msg = f"Question: {query}\n\n" f"Retrieved documentation context:\n{context}"

    try:
        from openai import AsyncOpenAI

        model_name = os.getenv("OPENAI_COPILOT_MODEL", "gpt-4o-mini")
        client = AsyncOpenAI(api_key=api_key, timeout=60.0)
        resp = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=600,
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception as exc:
        logger.warning("chat: OpenAI call failed (%s); falling back to top chunk", exc)
        return None


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/query")
async def chat_query(body: ChatRequest):
    """
    RAG-backed chat endpoint.

    Request body: {"message": "your question here"}

    Response:
      - status: "ok"
      - reply: grounded answer (OpenAI when key is set, else top retrieved chunk)
      - sources: list of {source, header, score} for the retrieved chunks
      - generated_at: ISO-8601 UTC timestamp
    """
    message = body.message.strip()
    if not message:
        raise HTTPException(400, detail="Field 'message' is required and cannot be empty.")

    _ensure_loaded()

    chunks = search(message, _TOP_K)

    if not chunks:
        return {
            "status": "ok",
            "message_received": message,
            "reply": (
                "No relevant documentation was found for your query. "
                "Try rephrasing or ask about the system architecture, API endpoints, "
                "feature engineering, or model training."
            ),
            "sources": [],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # Attempt OpenAI; fall back to returning the top chunk as-is
    reply = await _openai_reply(message, chunks)
    if reply is None:
        best = chunks[0]
        reply = f"**{best['header']}**\n\n{best['text']}"

    sources = [
        {
            "source": c["source"],
            "header": c["header"],
            "score": round(c["score"], 4),
        }
        for c in chunks
    ]

    logger.info(
        "chat_query: message_len=%d chunks=%d openai=%s",
        len(message),
        len(chunks),
        "yes" if (os.getenv("OPENAI_API_KEY") or "").strip() else "no",
    )

    return {
        "status": "ok",
        "message_received": message,
        "reply": reply,
        "sources": sources,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
