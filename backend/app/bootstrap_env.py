"""Load project-root ``.env`` before other app imports (secrets stay out of git)."""

from __future__ import annotations

from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    pass
else:
    _root = Path(__file__).resolve().parents[2]
    load_dotenv(_root / ".env")
