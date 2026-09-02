"""
Test the refactored model: predict_from_history must produce bit-for-bit
identical results to predict_track_intensity for built-in storms.
Also tests validate_history and the upload pipeline.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import pandas as pd

# Test 1: Bit-for-bit regression check
print("=== Test 1: predict_from_history == predict_track_intensity ===")
from src.model import predict_track_intensity, predict_from_history, validate_history

# Load storms.csv
df = pd.read_csv("data/processed/storms.csv")

test_storms = ["2019116N02090", "2018279N11069", "2023129N08091"]
for sid in test_storms:
    via_contract = predict_track_intensity(sid, [24, 48, 72])
    storm_df = df[df["storm_id"] == sid].copy()
    via_history = predict_from_history(storm_df, [24, 48, 72])

    # Compare track
    for i in range(3):
        c_track = via_contract["track"][i]
        h_track = via_history["track"][i]
        assert c_track["lat"] == h_track["lat"], f"lat mismatch at lead {c_track['lead_h']}h"
        assert c_track["lon"] == h_track["lon"], f"lon mismatch at lead {c_track['lead_h']}h"

    # Compare intensity
    for i in range(3):
        c_int = via_contract["intensity"][i]
        h_int = via_history["intensity"][i]
        assert c_int["wind_kt"] == h_int["wind_kt"], f"wind mismatch at lead {c_int['lead_h']}h"
        assert c_int["pressure_hpa"] == h_int["pressure_hpa"], f"pres mismatch at lead {c_int['lead_h']}h"

    print(f"  {sid}: IDENTICAL")

# Test 2: validate_history
print("\n=== Test 2: validate_history ===")
# Good storm
ok, msg = validate_history(df[df["storm_id"] == "2019116N02090"].copy())
assert ok, f"Expected ok=True, got: {msg}"
print("  Valid storm: PASS")

# Too few observations
tiny_df = df[df["storm_id"] == "2019116N02090"].head(2).copy()
ok, msg = validate_history(tiny_df)
assert not ok
assert "at least" in msg.lower()
print(f"  Too few obs: PASS ({msg})")

# Empty df
ok, msg = validate_history(pd.DataFrame())
assert not ok
print(f"  Empty df: PASS ({msg})")

# Test 3: clean_ibtracs_dataframe on the raw file
print("\n=== Test 3: clean_ibtracs_dataframe ===")
from src.data_pipeline import clean_ibtracs_dataframe, validate_upload
raw_df = pd.read_csv("data/raw/ibtracs.NI.list.v04r01.csv", low_memory=False)
cleaned = clean_ibtracs_dataframe(raw_df)
assert "storm_id" in cleaned.columns
assert "lat" in cleaned.columns
assert len(cleaned) > 0
n_storms = cleaned["storm_id"].nunique()
print(f"  Cleaned {n_storms} storms, {len(cleaned)} rows: PASS")

# Test 4: validate_upload
print("\n=== Test 4: validate_upload ===")
ok, storms_df, err = validate_upload(raw_df)
assert ok, f"validate_upload failed: {err}"
print(f"  Full IBTrACS file: PASS ({storms_df['storm_id'].nunique()} storms)")
if err:
    print(f"  Warnings: {err}")

# Test 5: Upload path produces same result as built-in for known storm
print("\n=== Test 5: Upload path bit-for-bit match ===")
# Filter raw to just Fani
fani_raw = raw_df[raw_df["SID"] == "2019116N02090"].copy()
ok2, fani_cleaned, err2 = validate_upload(fani_raw)
assert ok2
fani_cleaned_no_name = fani_cleaned.drop(columns=["name"]) if "name" in fani_cleaned.columns else fani_cleaned
via_upload = predict_from_history(fani_cleaned_no_name, [24, 48, 72])
via_builtin = predict_track_intensity("2019116N02090", [24, 48, 72])

for i in range(3):
    assert via_upload["track"][i]["lat"] == via_builtin["track"][i]["lat"]
    assert via_upload["track"][i]["lon"] == via_builtin["track"][i]["lon"]
    assert via_upload["intensity"][i]["wind_kt"] == via_builtin["intensity"][i]["wind_kt"]
print("  Fani upload vs built-in: IDENTICAL")

print("\n=== ALL REGRESSION TESTS PASSED ===")
