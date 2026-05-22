"""
Alert dispatcher: Slack webhook, SMTP email, PagerDuty Events API v2.

Called after every alerts recompute in monitoring_service. Channels are read
from config (monitoring.alerts.channels). Secrets come from environment
variables only — never from config files.

Cooldown is enforced per-channel per-alert-type using timestamps stored in
the monitoring state's ``last_dispatched`` dict so it survives restarts.
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import ssl
import urllib.request
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALERT_LABELS: dict[str, str] = {
    "mae": "MAE threshold exceeded",
    "mape": "MAPE threshold exceeded",
    "drift": "Feature drift detected",
}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _now_utc().isoformat().replace("+00:00", "Z")


def _minutes_since(iso_ts: str) -> float:
    try:
        past = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        return (_now_utc() - past).total_seconds() / 60.0
    except Exception:
        return float("inf")


def _active_alerts(alerts: dict[str, bool]) -> list[str]:
    return [k for k, v in alerts.items() if v]


# ---------------------------------------------------------------------------
# Channel senders
# ---------------------------------------------------------------------------


def _send_slack(message: str) -> None:
    url = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    if not url:
        logger.warning("alert_dispatcher: SLACK_WEBHOOK_URL not set; skipping Slack alert")
        return
    payload = json.dumps({"text": message}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status not in (200, 204):
                logger.warning("alert_dispatcher: Slack responded %s", resp.status)
    except Exception as exc:
        logger.warning("alert_dispatcher: Slack send failed: %s", exc)


def _send_email(subject: str, body: str) -> None:
    host = os.environ.get("SMTP_HOST", "").strip()
    port_str = os.environ.get("SMTP_PORT", "587").strip()
    user = os.environ.get("SMTP_USER", "").strip()
    password = os.environ.get("SMTP_PASSWORD", "").strip()
    from_addr = os.environ.get("ALERT_EMAIL_FROM", "").strip()
    to_addr = os.environ.get("ALERT_EMAIL_TO", "").strip()

    missing = [k for k, v in [
        ("SMTP_HOST", host), ("SMTP_USER", user), ("SMTP_PASSWORD", password),
        ("ALERT_EMAIL_FROM", from_addr), ("ALERT_EMAIL_TO", to_addr),
    ] if not v]
    if missing:
        logger.warning("alert_dispatcher: email skipped — missing env vars: %s", ", ".join(missing))
        return

    try:
        port = int(port_str)
    except ValueError:
        logger.warning("alert_dispatcher: invalid SMTP_PORT %r; skipping email", port_str)
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP(host, port, timeout=15) as server:
            server.ehlo()
            server.starttls(context=ctx)
            server.login(user, password)
            server.sendmail(from_addr, [to_addr], msg.as_string())
        logger.info("alert_dispatcher: email sent to %s", to_addr)
    except Exception as exc:
        logger.warning("alert_dispatcher: email send failed: %s", exc)


def _send_pagerduty(summary: str, details: str) -> None:
    key = os.environ.get("PAGERDUTY_INTEGRATION_KEY", "").strip()
    if not key:
        logger.warning("alert_dispatcher: PAGERDUTY_INTEGRATION_KEY not set; skipping PagerDuty alert")
        return

    payload = {
        "routing_key": key,
        "event_action": "trigger",
        "payload": {
            "summary": summary,
            "source": "e2e-forecasting-monitoring",
            "severity": "warning",
            "custom_details": {"details": details},
        },
    }
    data = json.dumps(payload).encode()
    url = "https://events.pagerduty.com/v2/enqueue"
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status not in (200, 202):
                logger.warning("alert_dispatcher: PagerDuty responded %s", resp.status)
            else:
                logger.info("alert_dispatcher: PagerDuty event enqueued")
    except Exception as exc:
        logger.warning("alert_dispatcher: PagerDuty send failed: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dispatch_alerts(
    alerts: dict[str, bool],
    context: dict[str, Any],
    channels: list[str],
    last_dispatched: dict[str, Any],
    cooldown_minutes: float = 60.0,
) -> dict[str, Any]:
    """
    Fire alert notifications for every active alert that is past its cooldown.

    Returns an updated ``last_dispatched`` dict (caller must persist it).

    Parameters
    ----------
    alerts:
        Dict of alert_type -> bool (e.g. {"mae": True, "mape": False, "drift": True}).
    context:
        Extra info for the message body (mae value, mape, drift score, etc.).
    channels:
        List of enabled channel names: "slack", "email", "pagerduty".
    last_dispatched:
        Mapping of channel -> alert_type -> ISO timestamp of last send.
    cooldown_minutes:
        Suppress re-sends within this window.
    """
    active = _active_alerts(alerts)
    if not active or not channels:
        return last_dispatched

    updated = {ch: dict(v) for ch, v in last_dispatched.items()} if last_dispatched else {}

    for alert_type in active:
        label = _ALERT_LABELS.get(alert_type, alert_type)
        for channel in channels:
            ch_state = updated.setdefault(channel, {})
            last_ts = ch_state.get(alert_type, "")
            if last_ts and _minutes_since(last_ts) < cooldown_minutes:
                logger.debug(
                    "alert_dispatcher: [%s/%s] within cooldown (%.1f min remaining)",
                    channel,
                    alert_type,
                    cooldown_minutes - _minutes_since(last_ts),
                )
                continue

            msg = _build_message(alert_type, label, context)
            _send_to_channel(channel, label, msg)
            ch_state[alert_type] = _iso_now()
            logger.info("alert_dispatcher: dispatched %s via %s", alert_type, channel)

    return updated


def _build_message(alert_type: str, label: str, context: dict[str, Any]) -> str:
    lines = [f"[ALERT] {label}"]
    if alert_type == "mae":
        lines.append(f"  MAE: {context.get('mae', 'n/a')!r}  (threshold: {context.get('mae_threshold', 'n/a')!r})")
    elif alert_type == "mape":
        lines.append(f"  MAPE: {context.get('mape', 'n/a')!r}  (threshold: {context.get('mape_threshold', 'n/a')!r})")
    elif alert_type == "drift":
        lines.append(f"  Drift score: {context.get('drift_score', 'n/a')!r}  (threshold: {context.get('drift_threshold', 'n/a')!r})")
    lines.append(f"  Model version: {context.get('model_version', 'unknown')}")
    lines.append(f"  As of: {context.get('as_of', _iso_now())}")
    return "\n".join(lines)


def _send_to_channel(channel: str, subject: str, message: str) -> None:
    if channel == "slack":
        _send_slack(message)
    elif channel == "email":
        _send_email(f"[Monitoring Alert] {subject}", message)
    elif channel == "pagerduty":
        _send_pagerduty(subject, message)
    else:
        logger.warning("alert_dispatcher: unknown channel %r — skipped", channel)
