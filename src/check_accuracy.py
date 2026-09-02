"""
check_accuracy.py -- ChakraNetra backtest & accuracy verification

CONTRACT.md Function Contract 7.

Backtests the model against:
  - The built-in held-out test split (reproduces RESULTS.md exactly)
  - Any uploaded historical storm (same rolling-forward methodology)

Exposes run_backtest() / summarize() as plain functions so the dashboard
can call them directly in an "Accuracy" tab without shelling out.

Usage (CLI):
    python -m src.check_accuracy                 # built-in test set
    python -m src.check_accuracy path/to/file.csv  # uploaded file
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is importable
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.model import (
    _build_features,
    _MODELS,
    _MODEL_LOADED,
    FEATURE_COLS,
    LEAD_TIMES,
    TARGETS,
    load_models,
    predict_from_history,
    validate_history,
    MIN_HISTORY_OBS,
)

RESULTS_DIR = BASE_DIR / "results"


# --------------------------------------------------------------------------- #
# Haversine (same formula as evaluate.py)
# --------------------------------------------------------------------------- #

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------- #
# Core backtest
# --------------------------------------------------------------------------- #

def run_backtest(
    storms_df: pd.DataFrame,
    lead_times: list[int] | None = None,
    include_calibration: bool = True,
) -> dict:
    """
    Rolling-forward backtest on historical storms.

    For each storm, at every valid timestep t, predicts +24/+48/+72h using
    the history up to t, and scores against what actually happened.

    Args:
        storms_df: Cleaned DataFrame (CONTRACT.md schema: storm_id,
                   timestamp, lat, lon, wind_kt, pressure_hpa, basin)
        lead_times: Lead times to evaluate (default [24, 48, 72])
        include_calibration: Whether to also check calibration coverage

    Returns:
        dict with per-lead-time metrics and per-storm breakdowns
    """
    if lead_times is None:
        lead_times = [24, 48, 72]

    # Ensure models are loaded
    global _MODEL_LOADED
    if not _MODEL_LOADED:
        load_models()

    # Build features for the full dataset at once
    df = storms_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    featured = _build_features(df)

    # Optionally load calibration
    cal_func = None
    if include_calibration:
        try:
            from src.calibration import calibrate
            cal_func = calibrate
        except ImportError:
            pass

    results = {
        "lead_times": {},
        "storms": {},
        "meta": {
            "n_storms": df["storm_id"].nunique(),
            "storm_ids": sorted(df["storm_id"].unique().tolist()),
            "model_loaded": _MODEL_LOADED,
        },
    }

    # Per-lead-time accumulators
    for lead_h in lead_times:
        track_errors = []
        wind_errors = []
        pressure_errors = []
        # Calibration coverage counters
        track_in_cone = 0
        wind_in_interval = 0
        n_calibrated = 0

        storm_results = {}

        for sid, grp in featured.groupby("storm_id"):
            grp = grp.sort_values("timestamp").reset_index(drop=True)
            times = grp["timestamp"].values
            lead_td = np.timedelta64(lead_h, "h")

            storm_track_errs = []
            storm_wind_errs = []
            storm_pres_errs = []

            for i in range(len(grp)):
                future_time = times[i] + lead_td
                matches = grp[grp["timestamp"] == future_time]
                if len(matches) != 1:
                    continue

                # Need enough history up to step i
                if i < MIN_HISTORY_OBS - 1:
                    continue

                actual = matches.iloc[0]

                # Run prediction using history up to observation i
                history_slice = df[df["storm_id"] == sid].copy()
                history_slice["timestamp"] = pd.to_datetime(history_slice["timestamp"])
                history_slice = history_slice.sort_values("timestamp").reset_index(drop=True)
                # Only keep rows up to and including i
                history_slice = history_slice.iloc[:i + 1]

                ok, msg = validate_history(history_slice)
                if not ok:
                    continue

                raw_pred = predict_from_history(history_slice, [lead_h])

                # Extract predicted values
                if raw_pred["track"]:
                    pred_lat = raw_pred["track"][0]["lat"]
                    pred_lon = raw_pred["track"][0]["lon"]
                    err_km = haversine_km(
                        actual["lat"], actual["lon"], pred_lat, pred_lon
                    )
                    storm_track_errs.append(err_km)
                    track_errors.append(err_km)

                if raw_pred["intensity"]:
                    pred_wind = raw_pred["intensity"][0]["wind_kt"]
                    pred_pres = raw_pred["intensity"][0]["pressure_hpa"]
                    storm_wind_errs.append(abs(actual["wind_kt"] - pred_wind))
                    storm_pres_errs.append(abs(actual["pressure_hpa"] - pred_pres))
                    wind_errors.append(abs(actual["wind_kt"] - pred_wind))
                    pressure_errors.append(abs(actual["pressure_hpa"] - pred_pres))

                # Calibration coverage check
                if cal_func is not None:
                    try:
                        cal_pred = cal_func(raw_pred)
                        if cal_pred.get("track") and cal_pred.get("intensity"):
                            cal_track = cal_pred["track"][0]
                            cal_int = cal_pred["intensity"][0]

                            cone_upper = cal_track.get("cone_km_upper", 0)
                            if cone_upper > 0:
                                actual_err = haversine_km(
                                    actual["lat"], actual["lon"],
                                    cal_track["lat"], cal_track["lon"]
                                )
                                if actual_err <= cone_upper:
                                    track_in_cone += 1
                                n_calibrated += 1

                            interval = cal_int.get("interval_kt")
                            if interval:
                                if interval[0] <= actual["wind_kt"] <= interval[1]:
                                    wind_in_interval += 1
                    except Exception:
                        pass

            if storm_track_errs:
                storm_results[sid] = {
                    "n_samples": len(storm_track_errs),
                    "mean_track_error_km": round(float(np.mean(storm_track_errs)), 2),
                    "wind_mae_kt": round(float(np.mean(storm_wind_errs)), 2) if storm_wind_errs else None,
                    "pressure_mae_hpa": round(float(np.mean(storm_pres_errs)), 2) if storm_pres_errs else None,
                }

        lead_result = {
            "n_samples": len(track_errors),
            "mean_track_error_km": round(float(np.mean(track_errors)), 2) if track_errors else None,
            "median_track_error_km": round(float(np.median(track_errors)), 2) if track_errors else None,
            "wind_mae_kt": round(float(np.mean(wind_errors)), 2) if wind_errors else None,
            "pressure_mae_hpa": round(float(np.mean(pressure_errors)), 2) if pressure_errors else None,
        }

        if n_calibrated > 0:
            lead_result["track_coverage"] = round(track_in_cone / n_calibrated, 3)
            lead_result["wind_coverage"] = round(wind_in_interval / n_calibrated, 3)
            lead_result["n_calibrated_samples"] = n_calibrated

        results["lead_times"][f"+{lead_h}h"] = lead_result
        results["storms"].update(
            {f"{sid}_+{lead_h}h": v for sid, v in storm_results.items()}
        )

    return results


def summarize(results: dict) -> str:
    """Format backtest results as a readable text table."""
    lines = []
    lines.append("=" * 70)
    lines.append("ChakraNetra Backtest Results")
    lines.append("=" * 70)
    lines.append("")

    meta = results.get("meta", {})
    lines.append(f"Storms: {meta.get('n_storms', '?')} | "
                 f"Model loaded: {meta.get('model_loaded', '?')}")
    lines.append("")

    lines.append(f"{'Lead':>6s}  {'Samples':>8s}  {'Track(km)':>10s}  "
                 f"{'Wind(kt)':>9s}  {'Pres(hPa)':>10s}  {'TrackCov':>9s}  {'WindCov':>8s}")
    lines.append("-" * 70)

    for lead_key in sorted(results.get("lead_times", {}).keys()):
        m = results["lead_times"][lead_key]
        track = f"{m['mean_track_error_km']:.1f}" if m.get("mean_track_error_km") else "--"
        wind = f"{m['wind_mae_kt']:.1f}" if m.get("wind_mae_kt") else "--"
        pres = f"{m['pressure_mae_hpa']:.1f}" if m.get("pressure_mae_hpa") else "--"
        tcov = f"{m['track_coverage']:.1%}" if m.get("track_coverage") is not None else "--"
        wcov = f"{m['wind_coverage']:.1%}" if m.get("wind_coverage") is not None else "--"
        lines.append(f"{lead_key:>6s}  {m['n_samples']:>8d}  {track:>10s}  "
                     f"{wind:>9s}  {pres:>10s}  {tcov:>9s}  {wcov:>8s}")

    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def main():
    """Run backtest from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description="ChakraNetra accuracy backtest")
    parser.add_argument(
        "csv_path", nargs="?", default=None,
        help="Path to an IBTrACS CSV. If omitted, uses the built-in test split."
    )
    parser.add_argument(
        "--no-calibration", action="store_true",
        help="Skip calibration coverage check"
    )
    args = parser.parse_args()

    if args.csv_path:
        # User-provided file
        from src.data_pipeline import clean_ibtracs_dataframe
        print(f"Loading: {args.csv_path}")
        raw_df = pd.read_csv(args.csv_path, low_memory=False)
        storms_df = clean_ibtracs_dataframe(raw_df)
        if "name" in storms_df.columns:
            storms_df = storms_df.drop(columns=["name"])
        print(f"Cleaned: {storms_df['storm_id'].nunique()} storms, {len(storms_df)} rows")
    else:
        # Built-in test split
        from src.data_pipeline import load_processed_data
        from src.model import load_split
        df = load_processed_data()
        _, test_ids = load_split()
        storms_df = df[df["storm_id"].isin(test_ids)].copy()
        print(f"Built-in test set: {len(test_ids)} storms, {len(storms_df)} rows")

    t0 = time.time()
    results = run_backtest(
        storms_df,
        include_calibration=not args.no_calibration,
    )
    elapsed = time.time() - t0

    print(summarize(results))
    print(f"Backtest completed in {elapsed:.1f}s")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = RESULTS_DIR / "backtest_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
