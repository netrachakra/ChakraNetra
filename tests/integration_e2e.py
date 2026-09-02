"""
integration_e2e.py -- End-to-end integration test for ChakraNetra

Runs the full pipeline for multiple real storm_ids:
  data -> predict_track_intensity() -> calibrate() -> compute_risk()
  -> API response (via TestClient) -> verify all keys

Reports per-storm timing and flags any integration issues.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd


def run_direct_pipeline(storm_id, lead_times):
    """Run the full pipeline via direct imports."""
    from src.model import predict_track_intensity
    from src.calibration import calibrate
    from src.risk import compute_risk

    t0 = time.perf_counter()

    # Step 1: predict
    raw = predict_track_intensity(storm_id, lead_times)
    t1 = time.perf_counter()

    # Step 2: calibrate
    cal = calibrate(raw)
    t2 = time.perf_counter()

    # Step 3: risk (from strongest predicted wind)
    strongest = max(cal["intensity"], key=lambda p: p["wind_kt"])
    matching_track = next(
        (t for t in cal["track"] if t["lead_h"] == strongest["lead_h"]),
        cal["track"][0],
    )
    risk = compute_risk(strongest["wind_kt"], matching_track["lat"], matching_track["lon"])
    t3 = time.perf_counter()

    return {
        "prediction": raw,
        "calibrated": cal,
        "risk": risk,
        "timing": {
            "predict_ms": round((t1 - t0) * 1000, 1),
            "calibrate_ms": round((t2 - t1) * 1000, 1),
            "risk_ms": round((t3 - t2) * 1000, 1),
            "total_ms": round((t3 - t0) * 1000, 1),
        },
    }


def run_api_pipeline(storm_id, lead_times):
    """Run via the FastAPI TestClient."""
    from fastapi.testclient import TestClient
    from src.api import app

    client = TestClient(app)
    t0 = time.perf_counter()
    resp = client.post("/v1/predict", json={
        "storm_id": storm_id,
        "lead_times_hours": lead_times,
    })
    t1 = time.perf_counter()
    return {
        "status_code": resp.status_code,
        "body": resp.json(),
        "api_ms": round((t1 - t0) * 1000, 1),
    }


def validate_response(body, storm_id, lead_times):
    """Check all CONTRACT.md keys are present and correct types."""
    errors = []

    # Top-level keys
    if body.get("storm_id") != storm_id:
        errors.append(f"storm_id mismatch: expected {storm_id}, got {body.get('storm_id')}")

    if "track" not in body:
        errors.append("Missing 'track'")
    if "intensity" not in body:
        errors.append("Missing 'intensity'")
    if "model_version" not in body:
        errors.append("Missing 'model_version'")

    # Track points
    for i, pt in enumerate(body.get("track", [])):
        for key in ["lead_h", "lat", "lon"]:
            if key not in pt:
                errors.append(f"track[{i}] missing '{key}'")
        # Calibration keys
        if "cone_km_upper" not in pt and pt.get("cone_km_upper") is None:
            errors.append(f"track[{i}] missing 'cone_km_upper'")
        # Verify lat/lon are in NI basin range
        lat, lon = pt.get("lat", 0), pt.get("lon", 0)
        if lat and (lat < -10 or lat > 40):
            errors.append(f"track[{i}] lat={lat} outside NI range")
        if lon and (lon < 40 or lon > 120):
            errors.append(f"track[{i}] lon={lon} outside NI range")

    # Intensity points
    for i, pt in enumerate(body.get("intensity", [])):
        for key in ["lead_h", "wind_kt", "pressure_hpa"]:
            if key not in pt:
                errors.append(f"intensity[{i}] missing '{key}'")
        if "interval_kt" not in pt and pt.get("interval_kt") is None:
            errors.append(f"intensity[{i}] missing 'interval_kt'")
        # Verify wind is positive
        if pt.get("wind_kt", 0) <= 0:
            errors.append(f"intensity[{i}] wind_kt={pt.get('wind_kt')} <= 0")
        # Verify interval contains prediction
        interval = pt.get("interval_kt")
        if interval and len(interval) == 2:
            if not (interval[0] <= pt["wind_kt"] <= interval[1]):
                errors.append(f"intensity[{i}] wind_kt={pt['wind_kt']} outside interval {interval}")

    # Empirical coverage
    cov = body.get("empirical_coverage")
    if cov == 0.8:
        errors.append("empirical_coverage is exactly 0.8 -- looks hardcoded!")

    # Risk
    risk = body.get("risk")
    if risk:
        if "risk_score" not in risk:
            errors.append("risk missing 'risk_score'")
        if "wind_radii_km" not in risk:
            errors.append("risk missing 'wind_radii_km'")
        else:
            for k in ["34kt", "50kt", "64kt"]:
                if k not in risk["wind_radii_km"]:
                    errors.append(f"risk.wind_radii_km missing '{k}'")
        # Verify risk_score bounds
        score = risk.get("risk_score", -1)
        if score < 0 or score > 1:
            errors.append(f"risk_score={score} outside [0,1]")

    return errors


def main():
    print("=" * 70)
    print("ChakraNetra End-to-End Integration Test")
    print("=" * 70)

    # Load storm IDs
    df = pd.read_csv(os.path.join("..", "data", "processed", "storms.csv")
                     if not os.path.exists("data/processed/storms.csv")
                     else "data/processed/storms.csv")
    all_storms = sorted(df["storm_id"].unique())
    print(f"\nTotal storms in dataset: {len(all_storms)}")

    # Pick 5 storms: mix of train and test
    test_storms_file = "data/processed/test_storm_ids.json"
    train_storms_file = "data/processed/train_storm_ids.json"
    with open(test_storms_file) as f:
        test_ids = json.load(f)
    with open(train_storms_file) as f:
        train_ids = json.load(f)

    # Use 2 test + 3 train storms for integration
    storms_to_test = test_ids[:2] + train_ids[:3]
    lead_times = [24, 48, 72]

    print(f"Testing {len(storms_to_test)} storms: {storms_to_test}")
    print(f"Lead times: {lead_times}")
    print()

    all_results = []
    all_errors = []

    for sid in storms_to_test:
        print(f"\n--- {sid} ---")

        # Direct pipeline
        try:
            direct = run_direct_pipeline(sid, lead_times)
            print(f"  Direct pipeline: {direct['timing']['total_ms']:.0f}ms "
                  f"(predict={direct['timing']['predict_ms']:.0f}ms, "
                  f"calibrate={direct['timing']['calibrate_ms']:.0f}ms, "
                  f"risk={direct['timing']['risk_ms']:.0f}ms)")
        except Exception as e:
            print(f"  DIRECT PIPELINE FAILED: {e}")
            all_errors.append((sid, "direct", str(e)))
            continue

        # API pipeline
        try:
            api = run_api_pipeline(sid, lead_times)
            print(f"  API pipeline:    {api['api_ms']:.0f}ms (status={api['status_code']})")
        except Exception as e:
            print(f"  API PIPELINE FAILED: {e}")
            all_errors.append((sid, "api", str(e)))
            api = None

        # Validate API response
        if api and api["status_code"] == 200:
            errors = validate_response(api["body"], sid, lead_times)
            if errors:
                for err in errors:
                    print(f"  VALIDATION ERROR: {err}")
                all_errors.append((sid, "validation", errors))
            else:
                print(f"  Validation: ALL CHECKS PASSED")

            # Quick summary
            body = api["body"]
            cov = body.get("empirical_coverage", "N/A")
            risk_score = body.get("risk", {}).get("risk_score", "N/A")
            print(f"  Coverage: {cov} | Risk: {risk_score} | Model: {body.get('model_version')}")

        all_results.append({
            "storm_id": sid,
            "direct_ms": direct["timing"]["total_ms"],
            "api_ms": api["api_ms"] if api else None,
            "api_status": api["status_code"] if api else None,
            "errors": len([e for e in all_errors if e[0] == sid]),
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    direct_times = [r["direct_ms"] for r in all_results]
    api_times = [r["api_ms"] for r in all_results if r["api_ms"]]
    print(f"Direct pipeline: avg {sum(direct_times)/len(direct_times):.0f}ms, "
          f"max {max(direct_times):.0f}ms")
    if api_times:
        print(f"API pipeline:    avg {sum(api_times)/len(api_times):.0f}ms, "
              f"max {max(api_times):.0f}ms")

    if not all_errors:
        print("\nINTEGRATION: ALL STORMS PASSED -- NO ERRORS")
    else:
        print(f"\nINTEGRATION: {len(all_errors)} ERROR(S) FOUND:")
        for sid, stage, err in all_errors:
            print(f"  {sid} @ {stage}: {err}")

    # Flag if too slow for live demo
    max_direct = max(direct_times) if direct_times else 0
    if max_direct > 5000:
        print(f"\nWARNING: Max direct pipeline time {max_direct:.0f}ms > 5s -- TOO SLOW FOR DEMO")
    elif max_direct > 2000:
        print(f"\nCAUTION: Max direct pipeline time {max_direct:.0f}ms > 2s -- borderline for demo")
    else:
        print(f"\nTIMING: OK for live demo (max {max_direct:.0f}ms)")

    return all_results, all_errors


if __name__ == "__main__":
    main()
