"""
Full-stack health check for the forecasting service.

Verifies the backend is reachable, model metadata is served correctly,
and the forecast endpoint returns a valid response.  Uses only the
requests library — no test framework required.

Usage (server must be running):
    python scripts/full_stack_check.py
    python scripts/full_stack_check.py --base-url http://127.0.0.1:8001
"""

import argparse
import sys
import time

import requests

STORE_ID = 1
HORIZON = 3


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-stack health check.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8001",
        help="Base URL of the running API server (default: http://127.0.0.1:8001)",
    )
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    passed = 0
    failed = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed, failed
        tag = "PASS" if ok else "FAIL"
        line = f"  [{tag}] {name}"
        if detail:
            line += f"  — {detail}"
        print(line)
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"Full-stack check against {base}\n")

    # ── 1. Backend liveness ──────────────────────────────────────────
    print("1. Backend liveness  (GET /health/live)")
    try:
        r = requests.get(f"{base}/health/live", timeout=5)
        check("server reachable", True)
        check("status 200", r.status_code == 200, f"got {r.status_code}")
    except requests.ConnectionError:
        check("server reachable", False, f"cannot connect to {base}")
        print(f"\nBackend is not running at {base}. Aborting.")
        sys.exit(1)

    # ── 2. Model info ────────────────────────────────────────────────
    print("\n2. Model metadata  (GET /api/v1/model/info)")
    r = requests.get(f"{base}/api/v1/model/info", timeout=10)
    check("status 200", r.status_code == 200, f"got {r.status_code}")
    if r.status_code == 200:
        body = r.json()
        check("has model_version", "model_version" in body)
        check("has feature_count", "feature_count" in body)
        check("has train_rmse", "train_rmse" in body)
        version = body.get("model_version", "?")
        features = body.get("feature_count", "?")
        print(f"         model_version={version}  features={features}")

    # ── 3. Forecast ──────────────────────────────────────────────────
    print(f"\n3. Forecast  (POST /api/v1/forecast/store  store_id={STORE_ID}, horizon={HORIZON})")
    t0 = time.perf_counter()
    r = requests.post(
        f"{base}/api/v1/forecast/store",
        json={"store_id": STORE_ID, "horizon": HORIZON},
        timeout=120,
    )
    latency_ms = round((time.perf_counter() - t0) * 1000)
    check("status 200", r.status_code == 200, f"got {r.status_code}")
    if r.status_code == 200:
        body = r.json()
        forecasts = body.get("forecasts", [])
        check(
            f"returned {HORIZON} forecasts",
            len(forecasts) == HORIZON,
            f"got {len(forecasts)}",
        )
        if forecasts:
            first = forecasts[0]
            check("forecast has 'date'", "date" in first)
            check("forecast has 'forecast'", "forecast" in first)
            check(
                "confidence bounds present",
                "confidence_low" in first and "confidence_high" in first,
            )
        print(f"         latency={latency_ms}ms")

    # ── Summary ──────────────────────────────────────────────────────
    total = passed + failed
    print(f"\n{'=' * 44}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed:
        print("FAILED — see details above")
        sys.exit(1)
    print("All full-stack checks passed")


if __name__ == "__main__":
    main()
