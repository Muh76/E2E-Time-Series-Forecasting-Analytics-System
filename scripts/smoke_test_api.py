"""
Lightweight smoke test for the forecasting API.

Calls core endpoints, validates status codes and response shape,
and prints a summary. Uses only the requests library — no pytest.

Usage (server must be running):
    python scripts/smoke_test_api.py
    python scripts/smoke_test_api.py --base-url http://localhost:8000
"""

import argparse
import sys

import requests

STORE_ID = 1
HORIZON = 7


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the forecasting API.")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the running API server (default: http://localhost:8000)",
    )
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    passed = 0
    failed = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed, failed
        status = "PASS" if ok else "FAIL"
        msg = f"  [{status}] {name}"
        if detail:
            msg += f"  — {detail}"
        print(msg)
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"Smoke testing API at {base}\n")

    # 1. GET /api/v1/model/info
    print("1. GET /api/v1/model/info")
    try:
        r = requests.get(f"{base}/api/v1/model/info", timeout=10)
        check("status 200", r.status_code == 200, f"got {r.status_code}")
        if r.status_code == 200:
            body = r.json()
            check("has model_version", "model_version" in body)
            check("has feature_count", "feature_count" in body)
    except requests.ConnectionError:
        check("connection", False, f"cannot reach {base}")
        print(f"\nServer not reachable at {base}. Is it running?")
        sys.exit(1)

    # 2. POST /api/v1/forecast/store
    print(f"\n2. POST /api/v1/forecast/store  (store_id={STORE_ID}, horizon={HORIZON})")
    r = requests.post(
        f"{base}/api/v1/forecast/store",
        json={"store_id": STORE_ID, "horizon": HORIZON},
        timeout=120,
    )
    check("status 200", r.status_code == 200, f"got {r.status_code}")
    if r.status_code == 200:
        body = r.json()
        forecasts = body.get("forecasts", [])
        check(
            f"forecast length == {HORIZON}",
            len(forecasts) == HORIZON,
            f"got {len(forecasts)}",
        )
        if forecasts:
            first = forecasts[0]
            check("forecast item has 'date'", "date" in first)
            check("forecast item has 'forecast'", "forecast" in first)

    # 2b. GET /api/v1/metrics (uses last forecast vs actuals when dates overlap)
    print("\n2b. GET /api/v1/metrics")
    r = requests.get(f"{base}/api/v1/metrics", timeout=10)
    check("status 200", r.status_code == 200, f"got {r.status_code}")
    if r.status_code == 200:
        body = r.json()
        check("has status", "status" in body)
        mst = body.get("status")
        check(
            "status is ok or no_ground_truth",
            mst in ("ok", "no_ground_truth"),
            f"got {mst}",
        )
        if mst == "ok":
            check("mae present", body.get("mae") is not None)
            check("rmse present", body.get("rmse") is not None)

    # 3. POST /api/v1/forecast/store/debug
    print(f"\n3. POST /api/v1/forecast/store/debug  (store_id={STORE_ID}, horizon={HORIZON})")
    r = requests.post(
        f"{base}/api/v1/forecast/store/debug",
        json={"store_id": STORE_ID, "horizon": HORIZON},
        timeout=10,
    )
    check("status 200", r.status_code == 200, f"got {r.status_code}")
    if r.status_code == 200:
        body = r.json()
        check("has last_observed_date", "last_observed_date" in body)
        check("has model_version", "model_version" in body)
        check(
            "recursive_steps matches horizon",
            body.get("recursive_steps") == HORIZON,
            f"got {body.get('recursive_steps')}",
        )

    # 4. GET /api/v1/monitoring/summary
    print("\n4. GET /api/v1/monitoring/summary")
    r = requests.get(f"{base}/api/v1/monitoring/summary", timeout=10)
    check("status 200", r.status_code == 200, f"got {r.status_code}")
    if r.status_code == 200:
        body = r.json()
        check("has performance", "performance" in body)
        check("has drift", "drift" in body)
        check("performance has mae", "mae" in body.get("performance", {}))

    # 5. GET /api/v1/monitoring/metrics
    print("\n5. GET /api/v1/monitoring/metrics")
    r = requests.get(f"{base}/api/v1/monitoring/metrics", timeout=10)
    check("status 200", r.status_code == 200, f"got {r.status_code}")
    if r.status_code == 200:
        body = r.json()
        check("has primary_metrics", "primary_metrics" in body)

    # 6. POST /api/v1/copilot/explain
    print("\n6. POST /api/v1/copilot/explain")
    r = requests.post(
        f"{base}/api/v1/copilot/explain",
        json={"query": "Summarize current model performance and drift."},
        timeout=30,
    )
    check("status 200", r.status_code == 200, f"got {r.status_code}")
    if r.status_code == 200:
        body = r.json()
        check("has explanation", "explanation" in body and len(body.get("explanation", "")) > 0)
        check("has sources", "sources" in body)

    # Summary
    total = passed + failed
    print(f"\n{'='*40}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("All API smoke tests passed")
    else:
        print("Some tests FAILED — see details above")
        sys.exit(1)


if __name__ == "__main__":
    main()
