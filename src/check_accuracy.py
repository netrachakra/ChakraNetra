"""
src/check_accuracy.py

Backtests ChakraNetra's prediction accuracy -- reproduces RESULTS.md's numbers
when run with no arguments, or scores a freshly uploaded IBTrACS CSV when
given --csv.

Two backtest modes:
  - FAST (default for dashboard): build features once per storm, vectorised
    inference at each timestep. ~100x faster than the slow path.
  - SLOW (--slow flag): calls predict_from_history() at every timestep,
    exercising the exact production code path. Use for regression testing.

CLI:
    python -m src.check_accuracy
    python -m src.check_accuracy --csv path/to/file.csv
    python -m src.check_accuracy --csv path/to/file.csv --storm-id 2023156N10067
    python -m src.check_accuracy --slow   # production-path regression test

Dashboard:
    from src.check_accuracy import render_accuracy_tab
    # inside dashboard/app.py, in a `with st.tab("Model Accuracy"):` block:
    render_accuracy_tab()
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_pipeline import clean_ibtracs_dataframe
from src.model import (
    _build_features,
    FEATURE_COLS,
    LEAD_TIMES,
    TARGETS,
    load_models,
    predict_from_history,
    validate_history,
    MIN_HISTORY_OBS,
)
# Import the module itself for mutable state access
import src.model as _model_mod

try:
    from src.calibration import calibrate
    HAS_CALIBRATION = True
except ImportError:
    HAS_CALIBRATION = False

LEAD_TIMES_H = [24, 48, 72]
EARTH_RADIUS_KM = 6371.0

# Max storms to backtest in dashboard upload mode (keeps it snappy)
MAX_UPLOAD_STORMS = 15


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km -- same metric RESULTS.md's track error uses."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _obs_step_hours(storm_df):
    """Infer observation cadence from real timestamps (usually 6h for IBTrACS)."""
    deltas = storm_df["timestamp"].diff().dropna()
    if deltas.empty:
        return 6.0
    return deltas.dt.total_seconds().median() / 3600.0


# --------------------------------------------------------------------------- #
# FAST backtest -- build features once, run inference from feature matrix
# --------------------------------------------------------------------------- #

def _ensure_models_loaded():
    """Load models once, return the dict."""
    if not _model_mod._MODEL_LOADED:
        load_models()
    return _model_mod._MODELS


def backtest_storm_fast(storm_df, lead_times_hours=LEAD_TIMES_H):
    """
    Vectorized rolling backtest for ONE storm.

    1. Build features once for the entire storm
    2. Collect all valid (i, j, lead_h) pairs via timestamp lookup
    3. Batch-predict all samples at once per model (12 batch calls total)
    4. Score against actuals

    ~20x faster than per-timestep predict_from_history calls.
    """
    models = _ensure_models_loaded()
    if not models:
        return pd.DataFrame()

    storm_df = storm_df.sort_values("timestamp").reset_index(drop=True)

    # Build features once for the entire storm
    featured = _build_features(storm_df)
    featured = featured.reset_index(drop=True)

    # Build timestamp -> row-index map
    time_to_idx = {}
    for r in range(len(featured)):
        time_to_idx[featured.iloc[r]["timestamp"]] = r

    # Collect all valid (i, j, lead_h) pairs
    n = len(featured)
    times = featured["timestamp"].values
    pairs = []  # list of (i, j, lead_h)

    for i in range(MIN_HISTORY_OBS - 1, n):
        t_i = times[i]
        for lead_h in lead_times_hours:
            target_time = t_i + np.timedelta64(lead_h, "h")
            j = time_to_idx.get(target_time)
            if j is not None:
                pairs.append((i, j, lead_h))

    if not pairs:
        return pd.DataFrame()

    # Extract feature matrix for all source indices at once
    source_indices = [p[0] for p in pairs]
    X_all = featured.iloc[source_indices][FEATURE_COLS].values.astype(np.float64)

    # Batch predict: one call per (target, lead_h) model
    # Group pairs by lead_h for batch inference
    from collections import defaultdict
    lead_groups = defaultdict(list)  # lead_h -> list of (pair_idx, i, j)
    for pair_idx, (i, j, lead_h) in enumerate(pairs):
        lead_groups[lead_h].append(pair_idx)

    # predictions[pair_idx] = {target: value}
    predictions = [{} for _ in pairs]

    for lead_h, pair_indices in lead_groups.items():
        X_batch = X_all[pair_indices]
        for target in TARGETS:
            key = (target, lead_h)
            if key not in models:
                nearest_h = min(LEAD_TIMES, key=lambda lh: abs(lh - lead_h))
                key = (target, nearest_h)
            if key in models:
                preds = models[key].predict(X_batch)
                for local_idx, pair_idx in enumerate(pair_indices):
                    predictions[pair_idx][target] = float(preds[local_idx])
            else:
                for pair_idx in pair_indices:
                    i = pairs[pair_idx][0]
                    predictions[pair_idx][target] = float(featured.iloc[i][target])

    # Score all pairs
    rows = []
    sid = storm_df["storm_id"].iloc[0]
    for pair_idx, (i, j, lead_h) in enumerate(pairs):
        actual = featured.iloc[j]
        pred = predictions[pair_idx]

        pred_lat = round(pred.get("lat", actual["lat"]), 2)
        pred_lon = round(pred.get("lon", actual["lon"]), 2)

        track_err = haversine_km(pred_lat, pred_lon, actual["lat"], actual["lon"])

        rows.append({
            "storm_id": sid,
            "lead_h": lead_h,
            "track_error_km": track_err,
            "wind_abs_err_kt": abs(round(pred.get("wind_kt", actual["wind_kt"]), 1) - actual["wind_kt"]),
            "pressure_abs_err_hpa": abs(round(pred.get("pressure_hpa", actual["pressure_hpa"]), 1) - actual["pressure_hpa"]),
        })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# SLOW backtest -- calls predict_from_history at every timestep (production path)
# --------------------------------------------------------------------------- #

def backtest_storm_slow(storm_df, lead_times_hours=LEAD_TIMES_H):
    """
    Slow rolling backtest for ONE storm. Calls predict_from_history() at
    every valid timestep -- exercises the exact production code path.
    Use for regression testing; use backtest_storm_fast() for dashboard.
    """
    storm_df = storm_df.sort_values("timestamp").reset_index(drop=True)
    step_h = _obs_step_hours(storm_df)
    rows = []

    for i in range(len(storm_df)):
        history = storm_df.iloc[: i + 1]
        ok, _ = validate_history(history)
        if not ok:
            continue

        raw_preds = predict_from_history(history, lead_times_hours)
        preds = calibrate(raw_preds) if HAS_CALIBRATION else raw_preds

        for lead_h in lead_times_hours:
            steps_ahead = round(lead_h / step_h)
            j = i + steps_ahead
            if j >= len(storm_df):
                continue

            actual = storm_df.iloc[j]
            actual_lead_h = (
                actual["timestamp"] - storm_df.iloc[i]["timestamp"]
            ).total_seconds() / 3600.0

            if abs(actual_lead_h - lead_h) > step_h / 2:
                continue

            track_pred = next(p for p in preds["track"] if p["lead_h"] == lead_h)
            intensity_pred = next(p for p in preds["intensity"] if p["lead_h"] == lead_h)

            track_err_km = haversine_km(
                track_pred["lat"], track_pred["lon"], actual["lat"], actual["lon"]
            )
            wind_err_kt = abs(intensity_pred["wind_kt"] - actual["wind_kt"])
            pressure_err_hpa = abs(intensity_pred["pressure_hpa"] - actual["pressure_hpa"])

            row = {
                "storm_id": storm_df["storm_id"].iloc[0],
                "lead_h": lead_h,
                "track_error_km": track_err_km,
                "wind_abs_err_kt": wind_err_kt,
                "pressure_abs_err_hpa": pressure_err_hpa,
            }

            if HAS_CALIBRATION and "cone_km_upper" in track_pred:
                row["track_covered"] = track_err_km <= track_pred["cone_km_upper"]
            if HAS_CALIBRATION and "interval_kt" in intensity_pred:
                lo, hi = intensity_pred["interval_kt"]
                row["wind_covered"] = lo <= actual["wind_kt"] <= hi

            rows.append(row)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def run_backtest(storms_df, storm_ids=None, lead_times_hours=LEAD_TIMES_H,
                 fast=True, max_storms=None, progress_callback=None):
    """
    Backtests every storm in storms_df (or just storm_ids if given).

    Args:
        fast: Use the fast path (default True). Set False for regression testing.
        max_storms: Limit to N storms (randomly sampled). None = all.
        progress_callback: callable(current, total) for progress updates.
    """
    _ensure_models_loaded()

    if storm_ids is not None:
        storms_df = storms_df[storms_df["storm_id"].isin(storm_ids)]

    unique_ids = sorted(storms_df["storm_id"].unique())

    # Limit storms if requested
    if max_storms and len(unique_ids) > max_storms:
        rng = np.random.default_rng(42)
        unique_ids = sorted(rng.choice(unique_ids, size=max_storms, replace=False))
        storms_df = storms_df[storms_df["storm_id"].isin(unique_ids)]

    backtest_fn = backtest_storm_fast if fast else backtest_storm_slow
    all_rows = []
    total = len(unique_ids)

    for idx, (sid, group) in enumerate(storms_df.groupby("storm_id")):
        if sid not in unique_ids:
            continue
        try:
            all_rows.append(backtest_fn(group, lead_times_hours))
        except Exception as e:
            print(f"  [skip] {sid}: {e}", file=sys.stderr)

        if progress_callback:
            progress_callback(idx + 1, total)

    cols = [
        "storm_id", "lead_h", "track_error_km",
        "wind_abs_err_kt", "pressure_abs_err_hpa",
    ]
    if not all_rows:
        return pd.DataFrame(columns=cols)
    return pd.concat(all_rows, ignore_index=True)


def summarize(results_df):
    """Aggregate per-lead-time metrics. Matches RESULTS.md table shape."""
    if results_df.empty:
        return pd.DataFrame()

    agg = {
        "track_error_km": ("track_error_km", "mean"),
        "wind_mae_kt": ("wind_abs_err_kt", "mean"),
        "pressure_mae_hpa": ("pressure_abs_err_hpa", "mean"),
        "samples": ("track_error_km", "count"),
    }
    if "track_covered" in results_df.columns:
        agg["track_coverage"] = ("track_covered", "mean")
    if "wind_covered" in results_df.columns:
        agg["wind_coverage"] = ("wind_covered", "mean")

    summary = (
        results_df.groupby("lead_h")
        .agg(**agg)
        .reset_index()
        .rename(columns={"lead_h": "lead_time_h"})
    )
    # Round for display
    for col in ["track_error_km", "wind_mae_kt", "pressure_mae_hpa"]:
        if col in summary.columns:
            summary[col] = summary[col].round(1)
    for col in ["track_coverage", "wind_coverage"]:
        if col in summary.columns:
            summary[col] = (summary[col] * 100).round(1)

    return summary


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Check ChakraNetra model accuracy")
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Raw IBTrACS CSV to backtest against. Omit to use the built-in "
             "held-out test split (reproduces RESULTS.md).",
    )
    parser.add_argument(
        "--storm-id", type=str, default=None,
        help="Restrict to one storm_id/SID.",
    )
    parser.add_argument(
        "--slow", action="store_true",
        help="Use the slow path (calls predict_from_history per timestep). "
             "Default is the fast path.",
    )
    parser.add_argument(
        "--out", type=str, default="results/accuracy_check.json",
        help="Where to write the JSON summary.",
    )
    args = parser.parse_args()

    if args.csv:
        raw_df = pd.read_csv(args.csv, low_memory=False)
        storms_df = clean_ibtracs_dataframe(raw_df)
        if "name" in storms_df.columns:
            storms_df = storms_df.drop(columns=["name"])
        storms_df["timestamp"] = pd.to_datetime(storms_df["timestamp"])
        print(f"Loaded {storms_df['storm_id'].nunique()} storm(s) from {args.csv}")
    else:
        storms_df = pd.read_csv("data/processed/storms.csv", parse_dates=["timestamp"])
        with open("data/processed/test_storm_ids.json") as f:
            test_ids = json.load(f)
        storms_df = storms_df[storms_df["storm_id"].isin(test_ids)]
        print(f"Loaded {len(test_ids)} held-out test storm(s) -- this reproduces RESULTS.md")

    storm_ids = [args.storm_id] if args.storm_id else None
    results_df = run_backtest(
        storms_df, storm_ids=storm_ids, fast=not args.slow,
    )

    if results_df.empty:
        print(
            f"No scoreable predictions -- every storm was shorter than "
            f"MIN_HISTORY_OBS ({MIN_HISTORY_OBS}) or had no ground truth far "
            f"enough ahead in the file."
        )
        sys.exit(1)

    summary = summarize(results_df)
    print("\n" + summary.to_string(index=False))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_json(args.out, orient="records", indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------- #
# Dashboard integration
# --------------------------------------------------------------------------- #

def render_accuracy_tab():
    """Call from inside a `with st.tab("Model Accuracy"):` block."""
    import streamlit as st

    st.subheader("Model Accuracy")

    mode = st.radio(
        "Backtest against",
        ["Built-in held-out storms", "Uploaded CSV"],
        horizontal=True,
    )

    if mode == "Built-in held-out storms":
        storms_df = pd.read_csv("data/processed/storms.csv", parse_dates=["timestamp"])
        with open("data/processed/test_storm_ids.json") as f:
            test_ids = json.load(f)
        target_df = storms_df[storms_df["storm_id"].isin(test_ids)]
        n_storms = len(test_ids)
        max_storms = None  # run all 5 test storms
    else:
        uploaded = st.file_uploader("Upload IBTrACS CSV", type="csv", key="accuracy_upload")
        if uploaded is None:
            st.info("Upload a file to run a fresh accuracy check.")
            return
        raw_df = pd.read_csv(uploaded, low_memory=False)
        target_df = clean_ibtracs_dataframe(raw_df)
        if "name" in target_df.columns:
            target_df = target_df.drop(columns=["name"])
        target_df["timestamp"] = pd.to_datetime(target_df["timestamp"])
        n_storms = target_df["storm_id"].nunique()

        # For large uploads, cap the number of storms
        if n_storms > MAX_UPLOAD_STORMS:
            st.info(
                f"File has {n_storms} storms. Will sample {MAX_UPLOAD_STORMS} "
                f"for speed. Use the CLI for a full run."
            )
        max_storms = MAX_UPLOAD_STORMS

    if st.button("Run accuracy check"):
        progress_bar = st.progress(0, text="Starting backtest...")
        storm_count = min(n_storms, max_storms) if max_storms else n_storms

        def _update_progress(current, total):
            pct = current / total
            progress_bar.progress(pct, text=f"Storm {current}/{total}...")

        results_df = run_backtest(
            target_df,
            fast=True,
            max_storms=max_storms,
            progress_callback=_update_progress,
        )
        progress_bar.progress(1.0, text="Done!")

        if results_df.empty:
            st.warning(
                "No storm in this file had enough history plus enough later "
                "observations to score. This needs complete historical "
                "storms, not an in-progress one."
            )
            return

        summary = summarize(results_df)
        st.dataframe(summary, use_container_width=True)

        # Per-storm breakdown
        if results_df["storm_id"].nunique() > 1:
            with st.expander("Per-storm breakdown"):
                per_storm = (
                    results_df.groupby(["storm_id", "lead_h"])
                    .agg(
                        track_km=("track_error_km", "mean"),
                        wind_kt=("wind_abs_err_kt", "mean"),
                        n=("track_error_km", "count"),
                    )
                    .reset_index()
                )
                per_storm["track_km"] = per_storm["track_km"].round(1)
                per_storm["wind_kt"] = per_storm["wind_kt"].round(1)
                st.dataframe(per_storm, use_container_width=True, hide_index=True)

        st.caption(
            "Track error = great-circle distance (km) between predicted and "
            "actual position. Wind/pressure = mean absolute error. Same "
            "metric definitions as RESULTS.md, so these numbers are directly "
            "comparable to the numbers on stage."
        )
