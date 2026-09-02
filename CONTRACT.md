# CHAKRANETRA PROTOTYPE — SHARED CONTRACT

> **Freeze this; do not change without telling the whole team.**

## GOAL

A working demo of AI-based cyclone track + intensity forecasting with calibrated
uncertainty and a wind-risk map, using real historical storm data, for a Sep 1 SIH
hackathon prototype demo.

No satellite imagery, no CNN, no Grad-CAM, no real SMS — those stay in the
Master Plan as designed-but-not-built for this sprint.

## REPO STRUCTURE

```
chakranetra-prototype/
  data/
    raw/                    # original IBTrACS download
    processed/
      storms.csv            # cleaned, one row per storm per timestamp
  src/
    data_pipeline.py        # loads/cleans data into storms.csv
    model.py                # predict_track_intensity()
    calibration.py          # calibrate()
    risk.py                 # compute_risk(), rankine_vortex()
    api.py                  # FastAPI app (optional — see 3-agent fallback)
  dashboard/
    app.py                  # Streamlit app
  tests/
  results/
    eval_metrics.json
    RESULTS.md
  README.md
  requirements.txt
```

## DATA SCHEMA — `data/processed/storms.csv`

| Column         | Type    | Example             |
|---------------|---------|---------------------|
| storm_id      | str     | "BOB-2023-04"       |
| timestamp     | ISO8601 | "2023-10-22T06:00:00" |
| lat           | float   | 13.5                |
| lon           | float   | 86.2                |
| wind_kt       | float   | 65.0                |
| pressure_hpa  | float   | 985.0               |
| basin         | str     | "BOB"               |

## FUNCTION CONTRACT 1 — `src/model.py`

```python
def predict_track_intensity(storm_id, lead_times_hours):
    # returns:
    # {
    #     "storm_id": str,
    #     "track": [{"lead_h": int, "lat": float, "lon": float}, ...],
    #     "intensity": [{"lead_h": int, "wind_kt": float, "pressure_hpa": float}, ...]
    # }
```

## FUNCTION CONTRACT 2 — `src/calibration.py`

```python
def calibrate(raw_prediction):
    # takes the exact dict shape returned by predict_track_intensity().
    # returns the SAME dict, with each track point given extra keys
    # "cone_km_lower" and "cone_km_upper", and each intensity point given
    # an extra key "interval_kt": [low, high]. Also adds a top-level
    # "empirical_coverage" float (e.g. 0.78) — the real measured coverage
    # on held-out data. Never hardcode this to exactly 0.8.
```

## FUNCTION CONTRACT 3 — `src/risk.py`

```python
def compute_risk(intensity_kt, lat, lon):
    # returns:
    # {
    #     "risk_score": float,
    #     "wind_radii_km": {"34kt": float, "50kt": float, "64kt": float}
    # }
```

## API CONTRACT (if built) — `POST /v1/predict`

- **Request**: `{"storm_id": "BOB-2023-04", "lead_times_hours": [24, 48, 72]}`
- **Response**: merges the three dicts above into one JSON object, plus `"model_version"`.

## RULE FOR ALL AGENTS

If a module you depend on isn't ready yet, build and commit a **MOCK** version
matching the exact contract shape above (clearly fake but plausible numbers) so
you are never blocked. Swap the mock for the real thing the moment it's pushed —
same function name, same signature, zero changes needed on your side if the
contract was followed.
