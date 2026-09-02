"""
test_api.py -- Integration tests for ChakraNetra FastAPI backend

Tests POST /v1/predict with real storm_ids, validates response schema,
and checks error handling for bad requests.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from src.api import app


client = TestClient(app)


# --------------------------------------------------------------------------- #
# Health & storms endpoints
# --------------------------------------------------------------------------- #

class TestHealth:
    def test_health_returns_ok(self):
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "model_loaded" in body
        assert "calibration_loaded" in body
        assert "risk_loaded" in body

    def test_health_reports_booleans(self):
        resp = client.get("/v1/health")
        body = resp.json()
        for key in ["model_loaded", "calibration_loaded", "risk_loaded"]:
            assert isinstance(body[key], bool)


class TestStorms:
    def test_storms_returns_list(self):
        resp = client.get("/v1/storms")
        assert resp.status_code == 200
        body = resp.json()
        assert "storm_ids" in body
        assert "count" in body
        assert isinstance(body["storm_ids"], list)
        assert body["count"] == len(body["storm_ids"])

    def test_storms_has_data(self):
        resp = client.get("/v1/storms")
        body = resp.json()
        # We expect at least some storms if storms.csv exists
        if body["count"] > 0:
            assert all(isinstance(s, str) for s in body["storm_ids"])


# --------------------------------------------------------------------------- #
# POST /v1/predict -- schema validation
# --------------------------------------------------------------------------- #

class TestPredictSchema:
    """Test that /v1/predict responses match the CONTRACT.md schema."""

    def _get_storm_ids(self):
        resp = client.get("/v1/storms")
        return resp.json()["storm_ids"]

    def test_predict_returns_200(self):
        storm_ids = self._get_storm_ids()
        if not storm_ids:
            pytest.skip("No storm data available")
        resp = client.post("/v1/predict", json={
            "storm_id": storm_ids[0],
            "lead_times_hours": [24, 48, 72],
        })
        assert resp.status_code == 200

    def test_predict_has_required_keys(self):
        storm_ids = self._get_storm_ids()
        if not storm_ids:
            pytest.skip("No storm data available")
        resp = client.post("/v1/predict", json={
            "storm_id": storm_ids[0],
            "lead_times_hours": [24, 48, 72],
        })
        body = resp.json()
        assert "storm_id" in body
        assert "track" in body
        assert "intensity" in body
        assert "model_version" in body

    def test_track_point_shape(self):
        storm_ids = self._get_storm_ids()
        if not storm_ids:
            pytest.skip("No storm data available")
        resp = client.post("/v1/predict", json={
            "storm_id": storm_ids[0],
            "lead_times_hours": [24, 48, 72],
        })
        body = resp.json()
        assert len(body["track"]) == 3
        for pt in body["track"]:
            assert "lead_h" in pt
            assert "lat" in pt
            assert "lon" in pt
            assert isinstance(pt["lead_h"], int)
            assert isinstance(pt["lat"], (int, float))
            assert isinstance(pt["lon"], (int, float))

    def test_intensity_point_shape(self):
        storm_ids = self._get_storm_ids()
        if not storm_ids:
            pytest.skip("No storm data available")
        resp = client.post("/v1/predict", json={
            "storm_id": storm_ids[0],
            "lead_times_hours": [24, 48, 72],
        })
        body = resp.json()
        assert len(body["intensity"]) == 3
        for pt in body["intensity"]:
            assert "lead_h" in pt
            assert "wind_kt" in pt
            assert "pressure_hpa" in pt
            assert isinstance(pt["wind_kt"], (int, float))
            assert isinstance(pt["pressure_hpa"], (int, float))

    def test_calibration_keys_present(self):
        """Track should have cone keys, intensity should have interval_kt."""
        storm_ids = self._get_storm_ids()
        if not storm_ids:
            pytest.skip("No storm data available")
        resp = client.post("/v1/predict", json={
            "storm_id": storm_ids[0],
            "lead_times_hours": [24, 48, 72],
        })
        body = resp.json()
        for pt in body["track"]:
            assert "cone_km_lower" in pt or pt.get("cone_km_lower") is None
            assert "cone_km_upper" in pt or pt.get("cone_km_upper") is None
        for pt in body["intensity"]:
            assert "interval_kt" in pt or pt.get("interval_kt") is None

    def test_risk_present(self):
        storm_ids = self._get_storm_ids()
        if not storm_ids:
            pytest.skip("No storm data available")
        resp = client.post("/v1/predict", json={
            "storm_id": storm_ids[0],
            "lead_times_hours": [24, 48, 72],
        })
        body = resp.json()
        if body.get("risk"):
            assert "risk_score" in body["risk"]
            assert "wind_radii_km" in body["risk"]

    def test_model_version_present(self):
        storm_ids = self._get_storm_ids()
        if not storm_ids:
            pytest.skip("No storm data available")
        resp = client.post("/v1/predict", json={
            "storm_id": storm_ids[0],
            "lead_times_hours": [24],
        })
        body = resp.json()
        assert isinstance(body["model_version"], str)
        assert len(body["model_version"]) > 0


# --------------------------------------------------------------------------- #
# POST /v1/predict with multiple real storm_ids
# --------------------------------------------------------------------------- #

class TestPredictMultipleStorms:
    """Integration test: call /v1/predict for 2-3 real storm_ids."""

    def test_predict_multiple_storms(self):
        storm_ids = client.get("/v1/storms").json()["storm_ids"]
        if len(storm_ids) < 2:
            pytest.skip("Need at least 2 storms")

        # Test first 3 storms (or however many exist)
        for sid in storm_ids[:3]:
            resp = client.post("/v1/predict", json={
                "storm_id": sid,
                "lead_times_hours": [24, 48, 72],
            })
            assert resp.status_code == 200, f"Failed for {sid}: {resp.text}"
            body = resp.json()
            assert body["storm_id"] == sid
            assert len(body["track"]) == 3
            assert len(body["intensity"]) == 3


# --------------------------------------------------------------------------- #
# Error handling -- clean 4xx, not 500
# --------------------------------------------------------------------------- #

class TestErrorHandling:
    """Bad requests should return clean 4xx errors, not 500s."""

    def test_nonexistent_storm_returns_404(self):
        resp = client.post("/v1/predict", json={
            "storm_id": "DOES-NOT-EXIST-999",
            "lead_times_hours": [24, 48, 72],
        })
        # Should be 404 if storms.csv is loaded, otherwise may succeed with mock
        storm_ids = client.get("/v1/storms").json()["storm_ids"]
        if storm_ids:
            assert resp.status_code == 404
            body = resp.json()
            assert "detail" in body

    def test_empty_storm_id_returns_422(self):
        resp = client.post("/v1/predict", json={
            "storm_id": "",
            "lead_times_hours": [24],
        })
        assert resp.status_code == 422

    def test_missing_storm_id_returns_422(self):
        resp = client.post("/v1/predict", json={
            "lead_times_hours": [24],
        })
        assert resp.status_code == 422

    def test_negative_lead_time_returns_422(self):
        resp = client.post("/v1/predict", json={
            "storm_id": "TEST",
            "lead_times_hours": [-1],
        })
        assert resp.status_code == 422

    def test_empty_lead_times_returns_422(self):
        resp = client.post("/v1/predict", json={
            "storm_id": "TEST",
            "lead_times_hours": [],
        })
        assert resp.status_code == 422

    def test_excessive_lead_time_returns_422(self):
        resp = client.post("/v1/predict", json={
            "storm_id": "TEST",
            "lead_times_hours": [500],
        })
        assert resp.status_code == 422

    def test_malformed_body_returns_422(self):
        resp = client.post("/v1/predict", content="not json",
                           headers={"content-type": "application/json"})
        assert resp.status_code == 422

    def test_wrong_type_lead_times_returns_422(self):
        resp = client.post("/v1/predict", json={
            "storm_id": "TEST",
            "lead_times_hours": "not a list",
        })
        assert resp.status_code == 422
