"""
src/check_accuracy.py

Backtests ChakraNetra's prediction accuracy -- reproduces RESULTS.md's numbers
when run with no arguments, or scores a freshly uploaded IBTrACS CSV when
given --csv.  Depends on the refactor in AGENT_BRIEF_csv_upload.md:

  - src.data_pipeline.clean_ibtracs_dataframe(raw_df) -> storms_df
  - src.model.predict_from_history(history_df, lead_times_hours)
  - src.model.validate_history(history_df) -> (bool, str)
  - src.model.MIN_HISTORY_OBS

CLI:
    python -m src.check_accuracy
        Backtests against the built-in held-out test split.  Numbers in the
        printed table should match RESULTS.md exactly -- use this as a
        regression check any time model.py or data_pipeline.py changes.

    python -m src.check_accuracy --csv path/to/some_export.csv
        Backtests against any IBTrACS-format file.  Only scores storms that
        are fully historical (i.e. the file has observations far enough past
        each forecast point to know what "actually happened").

    python -m src.check_accuracy --csv path/to/export.csv --storm-id 2023156N10067
        Restrict to one storm.

Dashboard:
    from src.check_accuracy import run_backtest, summarize, render_accuracy_tab
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
from src.model import predict_from_history, validate_history, MIN_HISTORY_OBS

try:
    from src.calibration import calibrate
    HAS_CALIBRATION = True
except ImportError:
    HAS_CALIBRATION = False

LEAD_TIMES_H = [24, 48, 72]
EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km -- same metric RESULTS.md's track error uses."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _obs_step_hours(storm_df):
    """Infer this storm's observation cadence from its real timestamps
    (usually 6h for IBTrACS, but don't hardcode it -- some exports differ)."""
    deltas = storm_df["timestamp"].diff().dropna()
    if deltas.empty:
        return 6.0
    return deltas.dt.total_seconds().median() / 3600.0


def backtest_storm(storm_df, lead_times_hours=LEAD_TIMES_H):
    """
    Rolling backtest for ONE storm.  At every valid timestamp t, feed the
    model everything up to and including t, predict t+lead, and compare
    against what actually happened at t+lead -- which we already know,
    because this is historical data, not a live storm.
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
                continue  # no ground truth that far ahead in this file yet

            actual = storm_df.iloc[j]
            actual_lead_h = (
                actual["timestamp"] - storm_df.iloc[i]["timestamp"]
            ).total_seconds() / 3600.0

            # only score if the actual observation lines up with the
            # intended lead time -- skip if the file has a gap here
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


def run_backtest(storms_df, storm_ids=None, lead_times_hours=LEAD_TIMES_H):
    """Backtests every storm in storms_df (or just storm_ids if given).

    NOTE: this calls the real model once per valid timestep per storm -- it's
    slow by design (~150ms/call per PIPELINE_STATUS.md), because it exercises
    the exact same code path as a live prediction rather than a vectorized
    shortcut that could drift from what the dashboard actually does."""
    if storm_ids is not None:
        storms_df = storms_df[storms_df["storm_id"].isin(storm_ids)]

    all_rows = []
    for sid, group in storms_df.groupby("storm_id"):
        try:
            all_rows.append(backtest_storm(group, lead_times_hours))
        except Exception as e:
            print(f"  [skip] {sid}: {e}", file=sys.stderr)

    cols = [
        "storm_id", "lead_h", "track_error_km",
        "wind_abs_err_kt", "pressure_abs_err_hpa",
    ]
    if not all_rows:
        return pd.DataFrame(columns=cols)
    return pd.concat(all_rows, ignore_index=True)


def summarize(results_df):
    """Matches the exact table shape in RESULTS.md (plus a coverage table
    if calibration is wired in)."""
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
    results_df = run_backtest(storms_df, storm_ids=storm_ids)

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
# Dashboard integration -- import this into dashboard/app.py
# --------------------------------------------------------------------------- #

def render_accuracy_tab():
    """Call from inside a `with st.tab("Model Accuracy"):` block in the
    Streamlit dashboard."""
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

    if st.button("Run accuracy check"):
        with st.spinner(
            "Backtesting -- this calls the real model once per timestep, "
            "so it can take a minute on larger files..."
        ):
            results_df = run_backtest(target_df)

        if results_df.empty:
            st.warning(
                "No storm in this file had enough history plus enough later "
                "observations to score. This needs complete historical "
                "storms, not an in-progress one."
            )
            return

        summary = summarize(results_df)
        st.dataframe(summary, use_container_width=True)

        st.caption(
            "Track error = great-circle distance (km) between predicted and "
            "actual position. Wind/pressure = mean absolute error. Same "
            "metric definitions as RESULTS.md, so these numbers are directly "
            "comparable to the numbers on stage."
        )
