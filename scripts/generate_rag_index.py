"""
Build the FAISS vector index and chunk metadata for the RAG chat endpoint.

Sources ingested:
  - README.md
  - docs/ARCHITECTURE.md, docs/API_CONTRACT.md, docs/DATA_CONTRACT.md
  - data/etl/README.md, config/README.md
  - config/base/default.yaml
  - artifacts/models/model_metadata.json, metrics.json, feature_columns.json

Outputs (written to data/):
  - data/faiss_index.bin       — FAISS IndexFlatIP (cosine, L2-normalised embeddings)
  - data/chunk_metadata.pkl    — list[dict] with keys: source, header, text

Usage:
    python scripts/generate_rag_index.py

Re-run whenever documentation or model metadata changes to refresh the index.
"""

from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"  WARNING: cannot read {path}: {exc}", file=sys.stderr)
        return ""


def _json_to_text(path: Path, title: str) -> str:
    """Convert a JSON file to a human-readable markdown-like block."""
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception as exc:
        print(f"  WARNING: cannot parse {path}: {exc}", file=sys.stderr)
        return ""

    lines = [f"# {title}"]
    items = data.items() if isinstance(data, dict) else enumerate(data)
    for k, v in items:
        if isinstance(v, (dict, list)):
            lines.append(f"{k}: {json.dumps(v, indent=2)}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _flush_section(
    chunks: list[dict],
    header: str,
    body_parts: list[str],
    source: str,
    max_chars: int,
) -> None:
    body = "\n".join(body_parts).strip()
    if not body:
        return

    full = f"{header}\n\n{body}"
    if len(full) <= max_chars:
        chunks.append({"source": source, "header": header, "text": full})
        return

    # Subdivide on blank lines to stay under max_chars
    paras = re.split(r"\n\n+", body)
    buf: list[str] = []
    buf_len = 0
    for para in paras:
        if buf_len + len(para) > max_chars and buf:
            chunks.append(
                {
                    "source": source,
                    "header": header,
                    "text": f"{header}\n\n" + "\n\n".join(buf),
                }
            )
            buf = []
            buf_len = 0
        buf.append(para)
        buf_len += len(para) + 2
    if buf:
        chunks.append(
            {
                "source": source,
                "header": header,
                "text": f"{header}\n\n" + "\n\n".join(buf),
            }
        )


def chunk_text(text: str, source: str, max_chars: int = 1000) -> list[dict]:
    """
    Split markdown on H1/H2/H3 headings.  Each heading starts a new chunk;
    oversized sections are split further on paragraph boundaries.
    """
    chunks: list[dict] = []
    heading_re = re.compile(r"^(#{1,3}\s+.+)$", re.MULTILINE)
    parts = heading_re.split(text)

    header = source
    body: list[str] = []

    for part in parts:
        if heading_re.match(part.strip()):
            _flush_section(chunks, header, body, source, max_chars)
            header = part.strip()
            body = []
        else:
            body.append(part)

    _flush_section(chunks, header, body, source, max_chars)
    return chunks


# ---------------------------------------------------------------------------
# Corpus collection
# ---------------------------------------------------------------------------


def collect_chunks(root: Path) -> list[dict]:
    chunks: list[dict] = []

    # Markdown documents
    md_files = [
        root / "README.md",
        root / "docs" / "ARCHITECTURE.md",
        root / "docs" / "API_CONTRACT.md",
        root / "docs" / "DATA_CONTRACT.md",
        root / "data" / "etl" / "README.md",
        root / "config" / "README.md",
    ]
    for path in md_files:
        if path.exists():
            text = _read_text(path)
            if text.strip():
                new = chunk_text(text, str(path.relative_to(root)))
                chunks.extend(new)
                print(f"  {path.relative_to(root)}: {len(new)} chunks")

    # YAML config as plain text (wrapped in a heading)
    yaml_path = root / "config" / "base" / "default.yaml"
    if yaml_path.exists():
        text = _read_text(yaml_path)
        if text.strip():
            new = chunk_text(f"# Default Configuration (YAML)\n\n{text}", str(yaml_path.relative_to(root)))
            chunks.extend(new)
            print(f"  {yaml_path.relative_to(root)}: {len(new)} chunks")

    # JSON model artifacts
    json_files = [
        (root / "artifacts" / "models" / "model_metadata.json", "Model Metadata"),
        (root / "artifacts" / "models" / "metrics.json", "Validation Metrics"),
        (root / "artifacts" / "models" / "feature_columns.json", "Feature Columns"),
    ]
    for path, title in json_files:
        if path.exists():
            text = _json_to_text(path, title)
            if text.strip():
                new = chunk_text(text, str(path.relative_to(root)))
                chunks.extend(new)
                print(f"  {path.relative_to(root)}: {len(new)} chunks")

    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def embed_texts(texts: list[str]) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "ERROR: sentence-transformers not installed.\n" "Run: pip install sentence-transformers",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Loading embedding model (all-MiniLM-L6-v2)…")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Embedding {len(texts)} chunks…")
    vecs = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,
    )
    return np.array(vecs, dtype="float32")


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------


def build_faiss_index(embeddings: np.ndarray):
    try:
        import faiss
    except ImportError:
        print(
            "ERROR: faiss-cpu not installed.\n" "Run: pip install faiss-cpu",
            file=sys.stderr,
        )
        sys.exit(1)

    dim = embeddings.shape[1]
    # IndexFlatIP = exact inner-product search; cosine similarity when embeddings are L2-normalised
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS IndexFlatIP built: {index.ntotal} vectors, dim={dim}")
    return index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"Project root: {ROOT}\n")

    print("Collecting document chunks…")
    chunks = collect_chunks(ROOT)
    if not chunks:
        print("ERROR: No chunks collected. Verify that doc files exist.", file=sys.stderr)
        sys.exit(1)
    print(f"Total: {len(chunks)} chunks\n")

    embeddings = embed_texts([c["text"] for c in chunks])

    index = build_faiss_index(embeddings)

    # Write outputs to data/
    out_dir = ROOT / "data"
    out_dir.mkdir(exist_ok=True)

    import faiss  # already imported in build_faiss_index; safe to re-import

    idx_path = out_dir / "faiss_index.bin"
    faiss.write_index(index, str(idx_path))
    print(f"\nSaved: {idx_path}")

    meta_path = out_dir / "chunk_metadata.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved: {meta_path} ({len(chunks)} chunks)")

    print("\nDone. Run the FastAPI backend and POST /api/v1/chat/query to test.")


if __name__ == "__main__":
    main()
