"""Verify lat/lon and pipeline output for 3 storms."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

df = pd.read_csv("data/processed/storms.csv")

# Test 3 storms
for sid in ["2019116N02090", "2021133N10071", "2023292N11063"]:
    sdf = df[df.storm_id == sid]
    lat_min, lat_max = sdf.lat.min(), sdf.lat.max()
    lon_min, lon_max = sdf.lon.min(), sdf.lon.max()
    basin = sdf.iloc[0].basin
    print(f"{sid} ({basin}): lat={lat_min:.1f}-{lat_max:.1f}, lon={lon_min:.1f}-{lon_max:.1f}")
    assert 0 < lat_max < 35, f"Lat out of NI range for {sid}"
    assert 40 < lon_max < 120, f"Lon out of NI range for {sid}"
    print(f"  -> Correct: North Indian Ocean")

# Pipeline output check
from src.model import predict_track_intensity
from src.calibration import calibrate
from src.risk import compute_risk

for sid in ["2019116N02090", "2018279N11069", "2023129N08091"]:
    raw = predict_track_intensity(sid, [24, 48, 72])
    cal = calibrate(raw)
    for pt in cal["track"]:
        assert 0 < pt["lat"] < 35, "Predicted lat out of range"
        assert 40 < pt["lon"] < 120, "Predicted lon out of range"
        assert pt.get("cone_km_upper") is not None, "Missing cone_km_upper"
    for pt in cal["intensity"]:
        assert pt.get("interval_kt") is not None, "Missing interval_kt"
    strongest = max(cal["intensity"], key=lambda p: p["wind_kt"])
    matching = next(t for t in cal["track"] if t["lead_h"] == strongest["lead_h"])
    risk = compute_risk(strongest["wind_kt"], matching["lat"], matching["lon"])
    assert 0 <= risk["risk_score"] <= 1
    assert "wind_radii_km" in risk
    print(f"{sid}: ALL pipeline outputs validated (track, intensity, cones, intervals, risk)")

print()
print("ALL VERIFICATIONS PASSED")
