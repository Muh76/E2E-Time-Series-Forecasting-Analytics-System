"""
Frontend package for the Time Series Forecasting & Analytics app.

Ensures the frontend directory is on sys.path so that imports like
`from components.api import ...` work when this package is imported
(e.g. from app.py as the entry point). Page files do not modify sys.path.
"""

import sys
from pathlib import Path

_FRONTEND_DIR = Path(__file__).resolve().parent
if str(_FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(_FRONTEND_DIR))
