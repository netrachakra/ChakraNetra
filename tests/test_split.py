"""
test_split.py — Tests for ChakraNetra train/test split integrity.

Critical test: ensures NO storm_id appears in both train and test sets.
This test must FAIL the build if any storm leaks between sets.
"""

import json
import os
import sys

import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestStormSplit:
    """Tests for storm-level train/test split."""

    def _load_split(self):
        base = os.path.join(os.path.dirname(__file__), "..")
        train_path = os.path.join(base, "data", "processed", "train_storm_ids.json")
        test_path = os.path.join(base, "data", "processed", "test_storm_ids.json")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            pytest.skip("Split files not yet generated — run model training first.")

        with open(train_path) as f:
            train_ids = json.load(f)
        with open(test_path) as f:
            test_ids = json.load(f)
        return train_ids, test_ids

    def test_no_storm_id_in_both_train_and_test(self):
        """CRITICAL: No storm_id may appear in both train and test sets."""
        train_ids, test_ids = self._load_split()
        overlap = set(train_ids) & set(test_ids)
        assert len(overlap) == 0, (
            f"DATA LEAK: {len(overlap)} storm(s) appear in BOTH train and test: {overlap}"
        )

    def test_split_is_nonempty(self):
        """Both train and test sets must have at least one storm."""
        train_ids, test_ids = self._load_split()
        assert len(train_ids) > 0, "Train set is empty!"
        assert len(test_ids) > 0, "Test set is empty!"

    def test_split_covers_all_storms(self):
        """Train + test should cover all storms in the dataset."""
        import pandas as pd
        base = os.path.join(os.path.dirname(__file__), "..")
        csv_path = os.path.join(base, "data", "processed", "storms.csv")

        if not os.path.exists(csv_path):
            pytest.skip("storms.csv not yet generated.")

        df = pd.read_csv(csv_path)
        all_storms = set(df["storm_id"].unique())

        train_ids, test_ids = self._load_split()
        split_storms = set(train_ids) | set(test_ids)

        assert all_storms == split_storms, (
            f"Split doesn't cover all storms. "
            f"Missing: {all_storms - split_storms}, "
            f"Extra: {split_storms - all_storms}"
        )

    def test_predict_contract_shape(self):
        """predict_track_intensity must return the exact CONTRACT.md shape."""
        from src.model import predict_track_intensity

        result = predict_track_intensity("TEST-STORM", [24, 48, 72])

        # Top-level keys
        assert "storm_id" in result
        assert "track" in result
        assert "intensity" in result
        assert isinstance(result["track"], list)
        assert isinstance(result["intensity"], list)
        assert len(result["track"]) == 3
        assert len(result["intensity"]) == 3

        # Track point shape
        for pt in result["track"]:
            assert "lead_h" in pt
            assert "lat" in pt
            assert "lon" in pt
            assert isinstance(pt["lead_h"], int)
            assert isinstance(pt["lat"], (int, float))
            assert isinstance(pt["lon"], (int, float))

        # Intensity point shape
        for pt in result["intensity"]:
            assert "lead_h" in pt
            assert "wind_kt" in pt
            assert "pressure_hpa" in pt
            assert isinstance(pt["lead_h"], int)
            assert isinstance(pt["wind_kt"], (int, float))
            assert isinstance(pt["pressure_hpa"], (int, float))

    def test_schema_storms_csv(self):
        """storms.csv must match CONTRACT.md schema."""
        import pandas as pd
        base = os.path.join(os.path.dirname(__file__), "..")
        csv_path = os.path.join(base, "data", "processed", "storms.csv")

        if not os.path.exists(csv_path):
            pytest.skip("storms.csv not yet generated.")

        df = pd.read_csv(csv_path)
        expected_cols = ["storm_id", "timestamp", "lat", "lon", "wind_kt", "pressure_hpa", "basin"]
        assert list(df.columns) == expected_cols, f"Schema mismatch: {list(df.columns)}"

        # Type checks
        assert df["lat"].dtype in ["float64", "float32"]
        assert df["lon"].dtype in ["float64", "float32"]
        assert df["wind_kt"].dtype in ["float64", "float32"]
        assert df["pressure_hpa"].dtype in ["float64", "float32"]
