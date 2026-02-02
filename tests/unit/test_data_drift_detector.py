"""
Unit tests for DataDriftDetector.

- Identical distributions → no drift
- Shifted mean → drift detected
- Empty or too-small sample → handled gracefully
- Deterministic behavior
"""

import numpy as np
import pandas as pd
import pytest

from models.monitoring.drift import DataDriftDetector


# ---------------------------------------------------------------------------
# Fixtures: small synthetic DataFrames
# ---------------------------------------------------------------------------


@pytest.fixture
def ref_df():
    """Reference: 20 values 1..20, uniform-ish."""
    return pd.DataFrame({"x": np.arange(1.0, 21.0), "y": np.arange(1.0, 21.0) * 2})


@pytest.fixture
def config_low_threshold():
    """Low threshold so shifted distribution triggers drift."""
    return {"monitoring": {"drift": {"threshold": 0.2, "n_bins": 5}}}


@pytest.fixture
def config_high_threshold():
    """High threshold so shifted distribution does not trigger (score max ~2)."""
    return {"monitoring": {"drift": {"threshold": 5.0, "n_bins": 5}}}


# ---------------------------------------------------------------------------
# Identical distributions → no drift
# ---------------------------------------------------------------------------


def test_identical_distributions_no_drift(ref_df, config_low_threshold):
    """Reference and current identical → overall_score ≈ 0, drift_detected False."""
    detector = DataDriftDetector(config_low_threshold)
    detector.fit_reference(ref_df)
    current_df = ref_df.copy()
    result = detector.detect_drift(current_df)
    assert result["overall_score"] == pytest.approx(0.0, abs=1e-10)
    assert result["drift_detected"] is False
    assert result["per_feature_scores"]["x"] == pytest.approx(0.0, abs=1e-10)
    assert result["per_feature_scores"]["y"] == pytest.approx(0.0, abs=1e-10)


def test_same_distribution_different_order_no_drift(ref_df, config_low_threshold):
    """Same values in different row order → no drift (deterministic binning)."""
    detector = DataDriftDetector(config_low_threshold)
    detector.fit_reference(ref_df)
    current_df = ref_df.sample(frac=1, random_state=42).reset_index(drop=True)
    result = detector.detect_drift(current_df)
    assert result["overall_score"] == pytest.approx(0.0, abs=1e-10)
    assert result["drift_detected"] is False


# ---------------------------------------------------------------------------
# Shifted mean → drift detected
# ---------------------------------------------------------------------------


def test_shifted_mean_drift_detected(ref_df, config_low_threshold):
    """Current distribution shifted (mean +50) → drift detected."""
    detector = DataDriftDetector(config_low_threshold)
    detector.fit_reference(ref_df)
    # Current: values 51..70 (shifted from 1..20); bins from ref are 1..20, so current all in last bin
    current_df = pd.DataFrame({
        "x": np.arange(51.0, 71.0),
        "y": np.arange(51.0, 71.0) * 2,
    })
    result = detector.detect_drift(current_df)
    assert result["overall_score"] > 0.2
    assert result["drift_detected"] is True


def test_shifted_mean_with_high_threshold_no_drift(ref_df, config_high_threshold):
    """Shifted distribution but threshold very high (5.0) → drift_detected False (score max ~2)."""
    detector = DataDriftDetector(config_high_threshold)
    detector.fit_reference(ref_df)
    current_df = pd.DataFrame({
        "x": np.arange(51.0, 71.0),
        "y": np.arange(51.0, 71.0) * 2,
    })
    result = detector.detect_drift(current_df)
    assert result["drift_detected"] is False
    assert result["overall_score"] <= 2.0


# ---------------------------------------------------------------------------
# Empty or too-small sample → handled gracefully
# ---------------------------------------------------------------------------


def test_empty_current_handled_gracefully(ref_df):
    """Current DataFrame with no rows → does not crash; returns valid structure."""
    detector = DataDriftDetector({"monitoring": {"drift": {"threshold": 0.25, "n_bins": 5}}})
    detector.fit_reference(ref_df)
    current_df = pd.DataFrame({"x": [], "y": []})
    result = detector.detect_drift(current_df)
    assert "overall_score" in result
    assert "drift_detected" in result
    assert "per_feature_scores" in result
    assert result["current_sample_size"] == 0
    # cur_hist all zeros → cur_prop all zeros → score = sum(ref_prop) = 1 per feature
    assert result["overall_score"] >= 0


def test_single_row_reference_handled_gracefully():
    """Reference with single row per feature → does not crash."""
    ref_df = pd.DataFrame({"x": [5.0], "y": [10.0]})
    detector = DataDriftDetector({"monitoring": {"drift": {"threshold": 0.25, "n_bins": 5}}})
    detector.fit_reference(ref_df)
    current_df = pd.DataFrame({"x": [5.0], "y": [10.0]})
    result = detector.detect_drift(current_df)
    assert result["overall_score"] >= 0
    assert "drift_detected" in result
    assert result["ref_sample_size"] == 1
    assert result["current_sample_size"] == 1


def test_small_sample_handled_gracefully():
    """Reference and current with few rows (e.g. 3) → runs without error."""
    ref_df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
    detector = DataDriftDetector({"monitoring": {"drift": {"threshold": 0.25, "n_bins": 3}}})
    detector.fit_reference(ref_df)
    current_df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
    result = detector.detect_drift(current_df)
    assert result["overall_score"] == pytest.approx(0.0, abs=1e-6)
    assert result["drift_detected"] is False


# ---------------------------------------------------------------------------
# Deterministic behavior
# ---------------------------------------------------------------------------


def test_deterministic_same_inputs_twice(ref_df, config_low_threshold):
    """Same inputs → identical output."""
    detector = DataDriftDetector(config_low_threshold)
    detector.fit_reference(ref_df)
    current_df = pd.DataFrame({"x": np.arange(51.0, 71.0), "y": np.arange(51.0, 71.0) * 2})
    r1 = detector.detect_drift(current_df)
    r2 = detector.detect_drift(current_df)
    assert r1["overall_score"] == r2["overall_score"]
    assert r1["per_feature_scores"] == r2["per_feature_scores"]
    assert r1["drift_detected"] == r2["drift_detected"]


def test_deterministic_fit_twice_same_result(ref_df):
    """Fit on same reference twice, then detect → same result."""
    config = {"monitoring": {"drift": {"threshold": 0.25, "n_bins": 5}}}
    d1 = DataDriftDetector(config)
    d1.fit_reference(ref_df)
    r1 = d1.detect_drift(ref_df.copy())

    d2 = DataDriftDetector(config)
    d2.fit_reference(ref_df)
    r2 = d2.detect_drift(ref_df.copy())

    assert r1["overall_score"] == r2["overall_score"]
    assert r1["drift_detected"] == r2["drift_detected"]


def test_fit_reference_before_detect_required():
    """detect_drift without fit_reference raises RuntimeError."""
    detector = DataDriftDetector({})
    with pytest.raises(RuntimeError, match="fit_reference"):
        detector.detect_drift(pd.DataFrame({"x": [1.0, 2.0]}))
