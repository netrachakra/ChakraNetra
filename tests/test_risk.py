"""
test_risk.py -- Unit tests for ChakraNetra risk module

Key test: monotonicity -- a stronger storm must NEVER produce a lower risk_score
at the same location.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.risk import (
    compute_risk,
    rankine_vortex,
    estimate_rmw_km,
    compute_wind_radii,
    _wind_risk_score,
)


class TestRiskMonotonicity:
    """A stronger storm must never produce a lower risk_score."""

    FIXED_LAT = 15.0
    FIXED_LON = 85.0

    def test_monotonicity_incremental(self):
        """risk_score must be non-decreasing as wind increases by 1 kt."""
        prev_score = -1.0
        for wind_kt in range(0, 180):
            result = compute_risk(float(wind_kt), self.FIXED_LAT, self.FIXED_LON)
            score = result["risk_score"]
            assert score >= prev_score, (
                f"MONOTONICITY VIOLATION: wind={wind_kt}kt -> score={score}, "
                f"but wind={wind_kt - 1}kt -> score={prev_score}"
            )
            prev_score = score

    def test_monotonicity_key_thresholds(self):
        """Test at Saffir-Simpson category boundaries."""
        thresholds = [25, 34, 50, 64, 83, 96, 113, 137, 160]
        scores = []
        for w in thresholds:
            result = compute_risk(float(w), self.FIXED_LAT, self.FIXED_LON)
            scores.append(result["risk_score"])

        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1], (
                f"MONOTONICITY VIOLATION at thresholds: "
                f"wind={thresholds[i]}kt (score={scores[i]}) < "
                f"wind={thresholds[i-1]}kt (score={scores[i-1]})"
            )

    def test_zero_wind_zero_risk(self):
        """Zero wind should give zero risk."""
        result = compute_risk(0.0, self.FIXED_LAT, self.FIXED_LON)
        assert result["risk_score"] == 0.0

    def test_extreme_wind_near_one(self):
        """Very strong wind should give risk close to 1.0."""
        result = compute_risk(170.0, self.FIXED_LAT, self.FIXED_LON)
        assert result["risk_score"] > 0.95

    def test_risk_score_bounded(self):
        """risk_score must always be in [0, 1]."""
        for wind_kt in range(0, 200):
            result = compute_risk(float(wind_kt), self.FIXED_LAT, self.FIXED_LON)
            assert 0.0 <= result["risk_score"] <= 1.0


class TestRankineVortex:
    """Tests for the modified Rankine vortex wind profile."""

    def test_zero_radius_zero_wind(self):
        assert rankine_vortex(0.0, 100.0) == 0.0

    def test_at_rmw_equals_vmax(self):
        """Wind at RMW should equal Vmax."""
        rmw = estimate_rmw_km(100.0)
        v = rankine_vortex(rmw, 100.0, rmw)
        assert abs(v - 100.0) < 0.01

    def test_inside_rmw_linear(self):
        """Inside RMW, wind should increase linearly."""
        rmw = 30.0
        vmax = 100.0
        v_half = rankine_vortex(rmw / 2, vmax, rmw)
        assert abs(v_half - vmax / 2) < 0.01

    def test_outside_rmw_decreasing(self):
        """Outside RMW, wind should decrease with distance."""
        rmw = 30.0
        vmax = 100.0
        v1 = rankine_vortex(rmw * 2, vmax, rmw)
        v2 = rankine_vortex(rmw * 3, vmax, rmw)
        assert v1 > v2

    def test_decay_exponent(self):
        """Outside RMW, verify the 0.5 exponent decay."""
        rmw = 30.0
        vmax = 100.0
        r = 60.0  # 2 * RMW
        expected = vmax * (rmw / r) ** 0.5
        actual = rankine_vortex(r, vmax, rmw)
        assert abs(actual - expected) < 0.01


class TestWindRadii:
    """Tests for wind radii computation."""

    def test_weak_storm_no_64kt_radius(self):
        """A 50kt storm should have zero 64kt radius."""
        radii = compute_wind_radii(50.0)
        assert radii["64kt"] == 0.0

    def test_strong_storm_has_all_radii(self):
        """A 130kt storm should have non-zero radii for all thresholds."""
        radii = compute_wind_radii(130.0)
        assert radii["34kt"] > 0
        assert radii["50kt"] > 0
        assert radii["64kt"] > 0

    def test_radii_ordering(self):
        """34kt radius > 50kt radius > 64kt radius."""
        radii = compute_wind_radii(100.0)
        assert radii["34kt"] > radii["50kt"] > radii["64kt"]


class TestComputeRiskContract:
    """Test that compute_risk() matches CONTRACT.md shape."""

    def test_output_shape(self):
        result = compute_risk(75.0, 15.0, 85.0)
        assert "risk_score" in result
        assert "wind_radii_km" in result
        assert isinstance(result["risk_score"], float)
        assert isinstance(result["wind_radii_km"], dict)
        assert "34kt" in result["wind_radii_km"]
        assert "50kt" in result["wind_radii_km"]
        assert "64kt" in result["wind_radii_km"]

    def test_all_values_are_float(self):
        result = compute_risk(100.0, 20.0, 80.0)
        assert isinstance(result["risk_score"], float)
        for key in ["34kt", "50kt", "64kt"]:
            assert isinstance(result["wind_radii_km"][key], float)


class TestRMWEstimate:
    """Tests for RMW regression."""

    def test_rmw_floor(self):
        """RMW should never go below 15 km."""
        rmw = estimate_rmw_km(200.0)
        assert rmw >= 15.0

    def test_rmw_decreases_with_intensity(self):
        """Stronger storms should have smaller RMW."""
        rmw_weak = estimate_rmw_km(40.0)
        rmw_strong = estimate_rmw_km(120.0)
        assert rmw_weak > rmw_strong
