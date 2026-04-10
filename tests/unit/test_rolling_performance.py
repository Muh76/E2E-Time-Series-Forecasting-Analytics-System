"""Unit tests for rolling performance JSON store."""

import pytest

from backend.app.services import rolling_performance as rp


@pytest.fixture
def isolated_store(monkeypatch, tmp_path):
    path = tmp_path / "rolling_performance.json"
    monkeypatch.setattr(rp, "_path", lambda: path)
    return path


def test_compute_rolling_empty(isolated_store):
    out = rp.compute_rolling_series(window=7, store_id=1)
    assert out["timestamps"] == []
    assert out["mae"] == []
    assert out["mape"] == []


def test_append_and_rolling_window(isolated_store):
    rp.append_evaluation_errors(
        1,
        ["2024-01-01", "2024-01-02", "2024-01-03"],
        [100.0, 100.0, 100.0],
        [90.0, 100.0, 110.0],
    )
    rp.append_evaluation_errors(
        1,
        ["2024-01-04", "2024-01-05"],
        [100.0, 100.0],
        [100.0, 100.0],
    )
    out = rp.compute_rolling_series(window=3, store_id=1)
    assert len(out["timestamps"]) >= 2
    assert len(out["mae"]) == len(out["timestamps"])
    assert len(out["mape"]) == len(out["timestamps"])
    # First window ends 2024-01-03: aes 10,0,10 -> mae 20/3
    assert out["timestamps"][0] == "2024-01-03"


def test_store_filter(isolated_store):
    rp.append_evaluation_errors(1, ["2024-06-01"], [50.0], [45.0])
    rp.append_evaluation_errors(2, ["2024-06-02"], [50.0], [40.0])
    out1 = rp.compute_rolling_series(window=2, store_id=1)
    out2 = rp.compute_rolling_series(window=2, store_id=2)
    assert len(out1["timestamps"]) == 0
    assert len(out2["timestamps"]) == 0
    rp.append_evaluation_errors(1, ["2024-06-02"], [50.0], [48.0])
    out1b = rp.compute_rolling_series(window=2, store_id=1)
    assert len(out1b["timestamps"]) == 1
