"""
evaluate.py -- ChakraNetra model evaluation

Evaluates the trained model on held-out test storms.
Reports:
  - Mean track error in km (haversine distance) per lead time
  - Intensity MAE (wind_kt) per lead time
  - Pressure MAE (pressure_hpa) per lead time

Outputs results/eval_metrics.json.
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.model import (
    FEATURE_COLS,
    LEAD_TIMES,
    TARGETS,
    _build_features,
    _build_training_samples,
    load_models,
    load_split,
    predict_track_intensity,
    _MODELS,
)
from src.data_pipeline import load_processed_data

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in km."""
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def evaluate_on_test_storms() -> dict:
    """
    Run evaluation on test storms.
    For each test storm, at each observation point, predict +24h/+48h/+72h
    and compare with actual future values.
    """
    df = load_processed_data()
    load_models()
    train_ids, test_ids = load_split()

    test_df = df[df["storm_id"].isin(test_ids)].copy()
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
    featured = _build_features(test_df)

    metrics = {}
    for lead_h in LEAD_TIMES:
        track_errors_km = []
        wind_errors_kt = []
        pressure_errors = []

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

                # Predict each target
                pred = {}
                for target in TARGETS:
                    key = (target, lead_h)
                    if key in _MODELS:
                        pred[target] = float(_MODELS[key].predict(x)[0])
                    else:
                        pred[target] = float(grp.iloc[i][target])

                # Track error (haversine)
                err_km = haversine_km(
                    actual["lat"], actual["lon"],
                    pred["lat"], pred["lon"]
                )
                track_errors_km.append(err_km)

                # Intensity errors
                wind_errors_kt.append(abs(actual["wind_kt"] - pred["wind_kt"]))
                pressure_errors.append(abs(actual["pressure_hpa"] - pred["pressure_hpa"]))

        metrics[f"+{lead_h}h"] = {
            "n_samples": len(track_errors_km),
            "mean_track_error_km": round(float(np.mean(track_errors_km)), 2) if track_errors_km else None,
            "median_track_error_km": round(float(np.median(track_errors_km)), 2) if track_errors_km else None,
            "wind_mae_kt": round(float(np.mean(wind_errors_kt)), 2) if wind_errors_kt else None,
            "pressure_mae_hpa": round(float(np.mean(pressure_errors)), 2) if pressure_errors else None,
        }

    # Summary
    metrics["test_storms"] = test_ids
    metrics["n_test_storms"] = len(test_ids)
    metrics["model_type"] = "HistGradientBoostingRegressor"
    metrics["note"] = "Evaluated on SYNTHETIC data -- not real IBTrACS. See README.md."

    return metrics


def write_metrics(metrics: dict):
    """Write metrics to results/eval_metrics.json."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = RESULTS_DIR / "eval_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote evaluation metrics -> {out_path}")


def write_results_md(metrics: dict):
    """Write RESULTS.md with honest summary."""
    lines = ["# ChakraNetra -- Evaluation Results\n"]
    lines.append("## Model: HistGradientBoostingRegressor (scikit-learn)\n")
    lines.append(f"Evaluated on **{metrics['n_test_storms']} held-out test storms** "
                 f"(synthetic data, not real IBTrACS).\n")
    lines.append("| Lead Time | Track Error (km) | Wind MAE (kt) | Pressure MAE (hPa) | Samples |")
    lines.append("|-----------|-----------------|---------------|--------------------|---------| ")

    for lead_h in LEAD_TIMES:
        key = f"+{lead_h}h"
        m = metrics.get(key, {})
        te = m.get("mean_track_error_km", "N/A")
        wm = m.get("wind_mae_kt", "N/A")
        pm = m.get("pressure_mae_hpa", "N/A")
        ns = m.get("n_samples", 0)
        lines.append(f"| +{lead_h}h | {te} | {wm} | {pm} | {ns} |")

    lines.append("\n## Honest Summary\n")
    lines.append(
        "This is a gradient-boosted baseline trained on ~15 training storms of **synthetic** "
        "cyclone data that mimics North Indian Ocean storm behavior. Track errors will likely "
        "be in the hundreds of km at +72h, which is expected for a non-ensemble, non-dynamical "
        "statistical model trained on a tiny dataset. Intensity MAE is similarly limited. "
        "These numbers establish a floor -- the model demonstrates the full pipeline works "
        "end-to-end (data -> features -> train -> predict -> evaluate) but is NOT competitive "
        "with operational forecasting. Real IBTrACS data and more storms would substantially "
        "improve results."
    )
    lines.append("\n> [!] **Synthetic data caveat**: All metrics above are on synthetic data. "
                 "They measure whether the model learns the patterns in our generated tracks, "
                 "not real-world cyclone predictability.\n")

    results_md_path = BASE_DIR / "RESULTS.md"
    with open(results_md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote results summary -> {results_md_path}")


if __name__ == "__main__":
    print("=== ChakraNetra Evaluation ===")
    metrics = evaluate_on_test_storms()
    print(json.dumps(metrics, indent=2))
    write_metrics(metrics)
    write_results_md(metrics)
