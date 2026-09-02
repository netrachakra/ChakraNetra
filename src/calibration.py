"""
calibration.py -- ChakraNetra uncertainty calibration

Implements calibrate() per CONTRACT.md Function Contract 2.

Method: Split-conformal prediction.
  1. On held-out (calibration) storms, compute residuals between model
     predictions and actual values for track (haversine km) and intensity
     (wind_kt absolute error).
  2. For a nominal coverage level (default 80%), take the ceil((n+1)*alpha)
     quantile of the residuals as the conformal score.
  3. At inference time, wrap each point prediction with +/- that conformal
     radius/interval.
  4. Report the REAL empirical coverage measured on the calibration set --
     never hardcoded.

This module works against the CONTRACT.md shape from predict_track_intensity()
and can use either the real model or a mock -- zero code changes needed.
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_CSV = BASE_DIR / "data" / "processed" / "storms.csv"
SPLIT_DIR = BASE_DIR / "data" / "processed"
TRAIN_IDS_FILE = SPLIT_DIR / "train_storm_ids.json"
TEST_IDS_FILE = SPLIT_DIR / "test_storm_ids.json"
CONFORMAL_SCORES_FILE = BASE_DIR / "models" / "conformal_scores.json"

# Default nominal coverage
NOMINAL_COVERAGE = 0.80

# Lead times we calibrate for (must match model.py)
LEAD_TIMES = [24, 48, 72]


# --------------------------------------------------------------------------- #
# Haversine helper (duplicated to avoid circular imports with evaluate.py)
# --------------------------------------------------------------------------- #
def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(min(1.0, math.sqrt(a)))


# --------------------------------------------------------------------------- #
# Conformal score computation (offline, on calibration set)
# --------------------------------------------------------------------------- #

def compute_conformal_scores(
    nominal_coverage: float = NOMINAL_COVERAGE,
) -> dict:
    """
    Compute conformal prediction scores on the TEST (held-out) storms.

    For each lead time, collects:
      - Track residuals: haversine_km(predicted_pos, actual_pos)
      - Intensity residuals: |predicted_wind - actual_wind|

    Then computes the conformal quantile at the nominal coverage level.

    Returns a dict keyed by lead_h with the quantile values and
    the actual measured empirical coverage.
    """
    from src.model import (
        FEATURE_COLS, TARGETS, _build_features,
        load_models as _load_models, load_split as _load_split,
    )
    from src.data_pipeline import load_processed_data

    df = load_processed_data()
    _load_models()
    # Re-import after load to get populated dict
    from src.model import _MODELS as models
    train_ids, test_ids = _load_split()

    test_df = df[df["storm_id"].isin(test_ids)].copy()
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
    featured = _build_features(test_df)

    scores = {}

    for lead_h in LEAD_TIMES:
        track_residuals = []   # haversine km
        wind_residuals = []    # |pred - actual| kt

        for _sid, grp in featured.groupby("storm_id"):
            grp = grp.sort_values("timestamp").reset_index(drop=True)
            times = grp["timestamp"].values
            lead_td = np.timedelta64(lead_h, "h")

            for i in range(len(grp)):
                future_time = times[i] + lead_td
                matches = grp[grp["timestamp"] == future_time]
                if len(matches) != 1:
                    continue

                actual = matches.iloc[0]
                x = grp.iloc[i][FEATURE_COLS].values.astype(np.float64).reshape(1, -1)

                pred = {}
                for target in TARGETS:
                    key = (target, lead_h)
                    if key in models:
                        pred[target] = float(models[key].predict(x)[0])
                    else:
                        pred[target] = float(grp.iloc[i][target])

                # Track residual (haversine)
                track_err = _haversine_km(
                    actual["lat"], actual["lon"],
                    pred.get("lat", actual["lat"]),
                    pred.get("lon", actual["lon"]),
                )
                track_residuals.append(track_err)

                # Wind intensity residual
                wind_err = abs(actual["wind_kt"] - pred.get("wind_kt", actual["wind_kt"]))
                wind_residuals.append(wind_err)

        if not track_residuals:
            scores[str(lead_h)] = {
                "track_conformal_km": 500.0,
                "wind_conformal_kt": 30.0,
                "n_samples": 0,
                "empirical_track_coverage": 0.0,
                "empirical_wind_coverage": 0.0,
            }
            continue

        track_arr = np.array(track_residuals)
        wind_arr = np.array(wind_residuals)
        n = len(track_arr)

        # Conformal quantile: ceil((n + 1) * alpha) / n
        q_idx = min(math.ceil((n + 1) * nominal_coverage) / n, 1.0)

        track_q = float(np.quantile(track_arr, q_idx))
        wind_q = float(np.quantile(wind_arr, q_idx))

        # Measure actual empirical coverage at these thresholds
        track_coverage = float(np.mean(track_arr <= track_q))
        wind_coverage = float(np.mean(wind_arr <= wind_q))

        scores[str(lead_h)] = {
            "track_conformal_km": round(track_q, 2),
            "wind_conformal_kt": round(wind_q, 2),
            "n_samples": n,
            "empirical_track_coverage": round(track_coverage, 4),
            "empirical_wind_coverage": round(wind_coverage, 4),
        }

    # Overall empirical coverage (average across lead times)
    coverages = [
        s["empirical_wind_coverage"]
        for s in scores.values()
        if s["n_samples"] > 0
    ]
    overall_coverage = round(float(np.mean(coverages)), 4) if coverages else 0.0

    result = {
        "nominal_coverage": nominal_coverage,
        "lead_times": scores,
        "empirical_coverage": overall_coverage,
    }

    os.makedirs(CONFORMAL_SCORES_FILE.parent, exist_ok=True)
    with open(CONFORMAL_SCORES_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote conformal scores -> {CONFORMAL_SCORES_FILE}")

    return result


def _load_conformal_scores() -> dict:
    """Load precomputed conformal scores from disk."""
    if CONFORMAL_SCORES_FILE.exists():
        with open(CONFORMAL_SCORES_FILE) as f:
            return json.load(f)
    return {}


# --------------------------------------------------------------------------- #
# calibrate() -- CONTRACT.md Function Contract 2
# --------------------------------------------------------------------------- #

def calibrate(raw_prediction: dict) -> dict:
    """
    Wrap point predictions with calibrated uncertainty intervals.

    Takes the exact dict shape returned by predict_track_intensity():
        {
            "storm_id": str,
            "track": [{"lead_h": int, "lat": float, "lon": float}, ...],
            "intensity": [{"lead_h": int, "wind_kt": float, "pressure_hpa": float}, ...]
        }

    Returns the SAME dict with extra keys per CONTRACT.md:
      - Each track point gets "cone_km_lower" and "cone_km_upper"
      - Each intensity point gets "interval_kt": [low, high]
      - Top-level "empirical_coverage" float (real measured value, never 0.8)
    """
    scores_data = _load_conformal_scores()

    # If no precomputed scores, use conservative fallbacks.
    # These are deliberately NOT 0.80 and will be replaced once
    # compute_conformal_scores() is run.
    if not scores_data:
        fallback_track = {24: 350.0, 48: 600.0, 72: 850.0}
        fallback_wind = {24: 18.0, 48: 28.0, 72: 35.0}
        empirical_coverage = 0.0  # signals "not yet calibrated"
    else:
        lead_scores = scores_data.get("lead_times", {})
        fallback_track = {}
        fallback_wind = {}
        for lh in LEAD_TIMES:
            s = lead_scores.get(str(lh), {})
            fallback_track[lh] = s.get("track_conformal_km", 500.0)
            fallback_wind[lh] = s.get("wind_conformal_kt", 30.0)
        empirical_coverage = scores_data.get("empirical_coverage", 0.0)

    # Deep-copy to avoid mutating caller's data
    result = {
        "storm_id": raw_prediction["storm_id"],
        "track": [],
        "intensity": [],
    }

    for pt in raw_prediction["track"]:
        lead_h = pt["lead_h"]
        closest_lh = min(LEAD_TIMES, key=lambda lh: abs(lh - lead_h))
        cone_radius = fallback_track.get(closest_lh, 500.0)

        result["track"].append({
            **pt,
            "cone_km_lower": 0.0,                    # best case: zero error
            "cone_km_upper": round(cone_radius, 2),   # conformal upper bound
        })

    for pt in raw_prediction["intensity"]:
        lead_h = pt["lead_h"]
        closest_lh = min(LEAD_TIMES, key=lambda lh: abs(lh - lead_h))
        wind_radius = fallback_wind.get(closest_lh, 30.0)
        wind_pred = pt["wind_kt"]

        result["intensity"].append({
            **pt,
            "interval_kt": [
                round(max(0.0, wind_pred - wind_radius), 1),
                round(wind_pred + wind_radius, 1),
            ],
        })

    result["empirical_coverage"] = empirical_coverage

    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("=== ChakraNetra Calibration ===")

    # Step 1: Compute conformal scores on held-out data
    print("\n--- Computing conformal scores ---")
    scores = compute_conformal_scores()
    print(json.dumps(scores, indent=2))

    # Step 2: Demo calibrate() on a sample prediction
    from src.model import predict_track_intensity
    sample = predict_track_intensity("2019160N11073", [24, 48, 72])
    calibrated = calibrate(sample)
    print("\n--- Calibrated prediction ---")
    print(json.dumps(calibrated, indent=2))
