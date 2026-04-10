"""
Persist per-date forecast errors for rolling MAE / MAPE monitoring.

Append on each successful evaluation vs ground truth; JSON file under
``data/monitoring/rolling_performance.json`` (override via env).
"""

from __future__ import annotations

import json
import logging
import math
import threading
from pathlib import Path
from typing import Any

from backend.app.runtime_paths import rolling_performance_json_path

logger = logging.getLogger(__name__)

_FILE_VERSION = 1
_MAX_POINTS = 5000
_LOCK = threading.Lock()


def _path() -> Path:
    return rolling_performance_json_path()


def _load_raw() -> dict[str, Any]:
    path = _path()
    if not path.exists():
        return {"version": _FILE_VERSION, "points": []}
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "points" not in data:
            return {"version": _FILE_VERSION, "points": []}
        pts = data.get("points") or []
        if not isinstance(pts, list):
            return {"version": _FILE_VERSION, "points": []}
        return {"version": _FILE_VERSION, "points": pts}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("rolling_performance: could not load %s: %s", path, exc)
        return {"version": _FILE_VERSION, "points": []}


def _save_raw(data: dict[str, Any]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=0)
    tmp.replace(path)


def append_evaluation_errors(
    store_id: int,
    dates: list[str],
    y_true: list[float],
    y_pred: list[float],
) -> None:
    """
    Upsert one row per (store_id, date) with absolute error and APE ratio.

    Called after a successful align of forecast vs actuals.
    """
    if len(dates) != len(y_true) or len(dates) != len(y_pred):
        logger.warning("rolling_performance: skip append (length mismatch) store_id=%s", store_id)
        return

    new_rows: list[dict[str, Any]] = []
    for dkey, yt, yp in zip(dates, y_true, y_pred):
        if not dkey or not math.isfinite(yt) or not math.isfinite(yp):
            continue
        ae = abs(float(yt) - float(yp))
        denom = abs(float(yt))
        ape = ae / denom if denom > 1e-12 else 0.0
        new_rows.append(
            {
                "date": str(dkey)[:10],
                "store_id": int(store_id),
                "abs_error": round(ae, 6),
                "ape": round(ape, 8),
            }
        )

    if not new_rows:
        return

    with _LOCK:
        data = _load_raw()
        points: list[dict[str, Any]] = data["points"]
        key_index: dict[tuple[int, str], int] = {}
        for i, p in enumerate(points):
            try:
                key_index[(int(p["store_id"]), str(p["date"])[:10])] = i
            except (KeyError, TypeError, ValueError):
                continue

        for row in new_rows:
            k = (row["store_id"], row["date"])
            if k in key_index:
                points[key_index[k]] = row
            else:
                points.append(row)
                key_index[k] = len(points) - 1

        points.sort(key=lambda x: (x.get("date", ""), x.get("store_id", 0)))
        if len(points) > _MAX_POINTS:
            points[:] = points[-_MAX_POINTS:]

        data["points"] = points
        total = len(points)
        try:
            _save_raw(data)
        except OSError as exc:
            logger.warning("rolling_performance: save failed: %s", exc)
            return

    logger.info(
        "rolling_performance: appended %d point(s) for store_id=%s (total=%d)",
        len(new_rows),
        store_id,
        total,
    )


def compute_rolling_series(
    window: int,
    store_id: int | None = None,
) -> dict[str, list[float] | list[str]]:
    """
    Rolling mean of abs_error (MAE) and mean APE over ``window`` consecutive
    points after sorting by date (then store_id).

    ``mape`` values in the response are **percent** (0–100 scale) for charting.

    Returns parallel lists ``timestamps``, ``mae``, ``mape``.
    """
    w = max(2, min(int(window), 90))
    data = _load_raw()
    pts: list[dict[str, Any]] = list(data.get("points") or [])

    if store_id is not None:
        sid = int(store_id)
        pts = [p for p in pts if int(p.get("store_id", -1)) == sid]

    pts.sort(key=lambda x: (str(x.get("date", "")), int(x.get("store_id", 0))))

    timestamps: list[str] = []
    mae_out: list[float] = []
    mape_pct_out: list[float] = []

    if len(pts) < w:
        return {"timestamps": timestamps, "mae": mae_out, "mape": mape_pct_out}

    for i in range(w - 1, len(pts)):
        lo = i - w + 1
        hi = i + 1
        chunk = pts[lo:hi]
        end_date = str(chunk[-1].get("date", ""))[:10]
        aes = [float(p["abs_error"]) for p in chunk if p.get("abs_error") is not None]
        apes = [float(p["ape"]) for p in chunk if p.get("ape") is not None]
        if not aes:
            continue
        timestamps.append(end_date)
        mae_out.append(round(sum(aes) / len(aes), 6))
        if apes:
            mape_pct_out.append(round(100.0 * sum(apes) / len(apes), 4))
        else:
            mape_pct_out.append(0.0)

    return {"timestamps": timestamps, "mae": mae_out, "mape": mape_pct_out}
