"""
check_coverage.py -- Empirical coverage check for calibration intervals

Runnable script that measures whether the calibrated intervals actually
contain the true value close to the nominal coverage rate (80%).

This is the source of truth for the "empirical_coverage" field in
calibrate()'s output -- never hardcoded.

Usage:
    python tests/check_coverage.py
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(min(1.0, math.sqrt(a)))


def run_coverage_check():
    """
    For each test storm, at each observation, predict forward at each
    lead time, calibrate the prediction, and check whether the true
    future value falls inside the calibrated interval.
    """
    from src.model import (
        FEATURE_COLS, TARGETS, LEAD_TIMES,
        _build_features, load_models, load_split,
    )
    from src.model import _MODELS as models_ref
    from src.data_pipeline import load_processed_data
    from src.calibration import calibrate, _load_conformal_scores

    # Load everything
    df = load_processed_data()
    load_models()
    from src.model import _MODELS as models
    train_ids, test_ids = load_split()

    scores_data = _load_conformal_scores()
    if not scores_data:
        print("ERROR: No conformal scores found. Run `python -m src.calibration` first.")
        sys.exit(1)

    test_df = df[df["storm_id"].isin(test_ids)].copy()
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
    featured = _build_features(test_df)

    print(f"Nominal coverage: {scores_data.get('nominal_coverage', 0.80)}")
    print(f"Test storms: {test_ids}")
    print()

    results = {}

    for lead_h in LEAD_TIMES:
        track_inside = 0
        track_total = 0
        wind_inside = 0
        wind_total = 0

        for sid, grp in featured.groupby("storm_id"):
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

                # Get point predictions
                pred = {}
                for target in TARGETS:
                    key = (target, lead_h)
                    if key in models:
                        pred[target] = float(models[key].predict(x)[0])
                    else:
                        pred[target] = float(grp.iloc[i][target])

                # Build a raw prediction in contract shape
                raw = {
                    "storm_id": sid,
                    "track": [{"lead_h": lead_h, "lat": pred["lat"], "lon": pred["lon"]}],
                    "intensity": [{"lead_h": lead_h, "wind_kt": pred["wind_kt"],
                                   "pressure_hpa": pred["pressure_hpa"]}],
                }

                # Calibrate
                cal = calibrate(raw)

                # Check track coverage: is actual within cone_km_upper?
                track_pt = cal["track"][0]
                actual_err = haversine_km(
                    track_pt["lat"], track_pt["lon"],
                    actual["lat"], actual["lon"],
                )
                if actual_err <= track_pt["cone_km_upper"]:
                    track_inside += 1
                track_total += 1

                # Check wind coverage: is actual within interval_kt?
                int_pt = cal["intensity"][0]
                interval = int_pt["interval_kt"]
                if interval[0] <= actual["wind_kt"] <= interval[1]:
                    wind_inside += 1
                wind_total += 1

        track_cov = track_inside / track_total if track_total > 0 else 0.0
        wind_cov = wind_inside / wind_total if wind_total > 0 else 0.0

        results[f"+{lead_h}h"] = {
            "track_coverage": round(track_cov, 4),
            "track_samples": track_total,
            "wind_coverage": round(wind_cov, 4),
            "wind_samples": wind_total,
        }

        print(f"+{lead_h}h: track coverage = {track_cov:.1%} ({track_inside}/{track_total}), "
              f"wind coverage = {wind_cov:.1%} ({wind_inside}/{wind_total})")

    # Overall
    all_wind_covs = [v["wind_coverage"] for v in results.values() if v["wind_samples"] > 0]
    all_track_covs = [v["track_coverage"] for v in results.values() if v["track_samples"] > 0]
    overall_wind = np.mean(all_wind_covs) if all_wind_covs else 0.0
    overall_track = np.mean(all_track_covs) if all_track_covs else 0.0

    print(f"\nOverall wind coverage:  {overall_wind:.1%}")
    print(f"Overall track coverage: {overall_track:.1%}")
    print(f"Reported empirical_coverage (from conformal scores): "
          f"{scores_data.get('empirical_coverage', 'N/A')}")

    # Verdict
    nominal = scores_data.get("nominal_coverage", 0.80)
    if overall_wind >= nominal - 0.05:
        print(f"\nVERDICT: Wind coverage ({overall_wind:.1%}) is within "
              f"tolerance of nominal ({nominal:.0%}). OK.")
    else:
        print(f"\nVERDICT: Wind coverage ({overall_wind:.1%}) is below "
              f"nominal ({nominal:.0%}). Intervals may be too narrow.")

    return results


if __name__ == "__main__":
    print("=== ChakraNetra Coverage Check ===\n")
    run_coverage_check()
