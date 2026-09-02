"""
test_calibration.py -- Tests for ChakraNetra calibration module

Tests that calibrate() returns the exact CONTRACT.md shape with
all required keys, and that empirical_coverage is never hardcoded to 0.8.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.calibration import calibrate


def _make_mock_prediction():
    """Create a mock prediction in CONTRACT.md shape."""
    return {
        "storm_id": "TEST-MOCK-01",
        "track": [
            {"lead_h": 24, "lat": 15.0, "lon": 85.0},
            {"lead_h": 48, "lat": 16.5, "lon": 84.0},
            {"lead_h": 72, "lat": 18.0, "lon": 83.0},
        ],
        "intensity": [
            {"lead_h": 24, "wind_kt": 65.0, "pressure_hpa": 985.0},
            {"lead_h": 48, "wind_kt": 55.0, "pressure_hpa": 990.0},
            {"lead_h": 72, "wind_kt": 45.0, "pressure_hpa": 995.0},
        ],
    }


class TestCalibrateContract:
    """Verify calibrate() output matches CONTRACT.md Function Contract 2."""

    def test_output_has_storm_id(self):
        raw = _make_mock_prediction()
        cal = calibrate(raw)
        assert cal["storm_id"] == "TEST-MOCK-01"

    def test_track_has_cone_keys(self):
        raw = _make_mock_prediction()
        cal = calibrate(raw)
        for pt in cal["track"]:
            assert "lead_h" in pt
            assert "lat" in pt
            assert "lon" in pt
            assert "cone_km_lower" in pt, "Missing cone_km_lower"
            assert "cone_km_upper" in pt, "Missing cone_km_upper"
            assert isinstance(pt["cone_km_lower"], (int, float))
            assert isinstance(pt["cone_km_upper"], (int, float))
            assert pt["cone_km_upper"] >= pt["cone_km_lower"]

    def test_intensity_has_interval_kt(self):
        raw = _make_mock_prediction()
        cal = calibrate(raw)
        for pt in cal["intensity"]:
            assert "lead_h" in pt
            assert "wind_kt" in pt
            assert "pressure_hpa" in pt
            assert "interval_kt" in pt, "Missing interval_kt"
            interval = pt["interval_kt"]
            assert isinstance(interval, list)
            assert len(interval) == 2
            assert interval[0] <= interval[1], "interval_kt[0] > interval_kt[1]"

    def test_has_empirical_coverage(self):
        raw = _make_mock_prediction()
        cal = calibrate(raw)
        assert "empirical_coverage" in cal
        assert isinstance(cal["empirical_coverage"], float)

    def test_empirical_coverage_not_hardcoded_08(self):
        """empirical_coverage must NEVER be exactly 0.8."""
        raw = _make_mock_prediction()
        cal = calibrate(raw)
        assert cal["empirical_coverage"] != 0.8, (
            "empirical_coverage is exactly 0.8 -- this looks hardcoded! "
            "Must be computed from real data."
        )

    def test_preserves_all_lead_times(self):
        raw = _make_mock_prediction()
        cal = calibrate(raw)
        assert len(cal["track"]) == 3
        assert len(cal["intensity"]) == 3
        lead_times = [pt["lead_h"] for pt in cal["track"]]
        assert lead_times == [24, 48, 72]

    def test_interval_contains_prediction(self):
        """The prediction itself should always be inside its own interval."""
        raw = _make_mock_prediction()
        cal = calibrate(raw)
        for pt in cal["intensity"]:
            interval = pt["interval_kt"]
            assert interval[0] <= pt["wind_kt"] <= interval[1], (
                f"Prediction {pt['wind_kt']} outside its own interval {interval}"
            )

    def test_cone_grows_with_lead_time(self):
        """Uncertainty cone should generally grow with lead time."""
        raw = _make_mock_prediction()
        cal = calibrate(raw)
        cones = [pt["cone_km_upper"] for pt in cal["track"]]
        # At minimum, +72h cone should be >= +24h cone
        assert cones[2] >= cones[0], (
            f"72h cone ({cones[2]}) smaller than 24h cone ({cones[0]})"
        )

    def test_does_not_mutate_input(self):
        """calibrate() should not modify the raw_prediction dict."""
        import copy
        raw = _make_mock_prediction()
        raw_copy = copy.deepcopy(raw)
        calibrate(raw)
        assert raw == raw_copy
