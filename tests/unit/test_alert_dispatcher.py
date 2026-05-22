"""
Tests for backend.app.services.alert_dispatcher.

Covers:
- No-op when channels list is empty.
- No-op when no alerts are active.
- dispatch_alerts() returns updated last_dispatched for active alerts.
- Cooldown suppresses re-dispatch within the window.
- Past-cooldown timestamps allow re-dispatch.
- Unknown channel is logged but does not raise.
- Message builder includes expected metric values.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from backend.app.services.alert_dispatcher import (
    _build_message,
    _minutes_since,
    dispatch_alerts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _past_iso(minutes_ago: float) -> str:
    return _iso(datetime.now(timezone.utc) - timedelta(minutes=minutes_ago))


_CONTEXT = {
    "mae": 18.5,
    "mape": 22.0,
    "drift_score": 0.30,
    "mae_threshold": 15.0,
    "mape_threshold": 20.0,
    "drift_threshold": 0.25,
    "model_version": "v3",
}


# ---------------------------------------------------------------------------
# No-op cases
# ---------------------------------------------------------------------------

def test_no_channels_returns_unchanged():
    alerts = {"mae": True, "mape": False, "drift": True}
    result = dispatch_alerts(alerts, _CONTEXT, channels=[], last_dispatched={}, cooldown_minutes=60)
    assert result == {}


def test_no_active_alerts_returns_unchanged():
    alerts = {"mae": False, "mape": False, "drift": False}
    result = dispatch_alerts(alerts, _CONTEXT, channels=["slack"], last_dispatched={}, cooldown_minutes=60)
    assert result == {}


# ---------------------------------------------------------------------------
# Dispatch fires and records timestamp
# ---------------------------------------------------------------------------

def test_dispatch_records_timestamp_for_active_alert():
    alerts = {"mae": True, "mape": False, "drift": False}
    with patch("backend.app.services.alert_dispatcher._send_slack") as mock_slack:
        result = dispatch_alerts(alerts, _CONTEXT, channels=["slack"], last_dispatched={}, cooldown_minutes=60)
    mock_slack.assert_called_once()
    assert "mae" in result.get("slack", {})


def test_dispatch_fires_per_active_alert_per_channel():
    alerts = {"mae": True, "mape": False, "drift": True}
    with patch("backend.app.services.alert_dispatcher._send_slack") as mock_slack:
        result = dispatch_alerts(alerts, _CONTEXT, channels=["slack"], last_dispatched={}, cooldown_minutes=60)
    assert mock_slack.call_count == 2
    assert "mae" in result["slack"]
    assert "drift" in result["slack"]


# ---------------------------------------------------------------------------
# Cooldown suppression
# ---------------------------------------------------------------------------

def test_cooldown_suppresses_recent_dispatch():
    alerts = {"mae": True, "mape": False, "drift": False}
    last_dispatched = {"slack": {"mae": _past_iso(10)}}  # 10 min ago < 60 min cooldown
    with patch("backend.app.services.alert_dispatcher._send_slack") as mock_slack:
        result = dispatch_alerts(alerts, _CONTEXT, channels=["slack"], last_dispatched=last_dispatched, cooldown_minutes=60)
    mock_slack.assert_not_called()
    # Timestamp should remain unchanged
    assert result["slack"]["mae"] == last_dispatched["slack"]["mae"]


def test_cooldown_allows_dispatch_after_window():
    alerts = {"mae": True, "mape": False, "drift": False}
    last_dispatched = {"slack": {"mae": _past_iso(90)}}  # 90 min ago > 60 min cooldown
    with patch("backend.app.services.alert_dispatcher._send_slack") as mock_slack:
        result = dispatch_alerts(alerts, _CONTEXT, channels=["slack"], last_dispatched=last_dispatched, cooldown_minutes=60)
    mock_slack.assert_called_once()
    # Timestamp should be updated to a newer value
    assert result["slack"]["mae"] != last_dispatched["slack"]["mae"]


# ---------------------------------------------------------------------------
# Unknown channel
# ---------------------------------------------------------------------------

def test_unknown_channel_does_not_raise():
    alerts = {"mae": True, "mape": False, "drift": False}
    result = dispatch_alerts(alerts, _CONTEXT, channels=["unknown_channel"], last_dispatched={}, cooldown_minutes=60)
    assert "mae" in result.get("unknown_channel", {})


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def test_build_message_mae_includes_values():
    msg = _build_message("mae", "MAE threshold exceeded", _CONTEXT)
    assert "18.5" in msg
    assert "15.0" in msg
    assert "v3" in msg


def test_build_message_mape_includes_values():
    msg = _build_message("mape", "MAPE threshold exceeded", _CONTEXT)
    assert "22.0" in msg
    assert "20.0" in msg


def test_build_message_drift_includes_values():
    msg = _build_message("drift", "Feature drift detected", _CONTEXT)
    assert "0.3" in msg
    assert "0.25" in msg


# ---------------------------------------------------------------------------
# _minutes_since helper
# ---------------------------------------------------------------------------

def test_minutes_since_recent():
    ts = _past_iso(5)
    assert abs(_minutes_since(ts) - 5.0) < 0.1


def test_minutes_since_invalid_returns_inf():
    assert _minutes_since("not-a-date") == float("inf")
