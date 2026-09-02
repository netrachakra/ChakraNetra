# ChakraNetra — AI Cyclone Forecasting Prototype

> SIH Hackathon Prototype — Sep 2026

## What This Is

ChakraNetra is a prototype AI system for tropical cyclone track and intensity
forecasting in the North Indian Ocean basin (Bay of Bengal + Arabian Sea).

This branch (`feature/data-model`) implements the data pipeline, baseline
forecasting model, and evaluation framework.

## Data Source

**Real IBTrACS data** — NOAA International Best Track Archive for Climate
Stewardship (IBTrACS), v04r01, North Indian Ocean basin.

- Source: https://www.ncei.noaa.gov/products/international-best-track-archive
- File: `data/raw/ibtracs.NI.list.v04r01.csv`
- 20 storms selected (2018–2023), 993 cleaned observations
- Mix of Bay of Bengal (BOB) and Arabian Sea (ARB) cyclones
- Includes major storms: Fani, Amphan, Tauktae, Biparjoy, Mocha, and others

### Selected Storms

| Storm SID | Approx. Name | Basin | Year | Observations |
|-----------|-------------|-------|------|-------------|
| 2019116N02090 | Fani | BOB | 2019 | 62 |
| 2020136N10088 | Amphan | BOB | 2020 | 44 |
| 2021133N10071 | Tauktae | ARB | 2021 | 40 |
| 2023156N10067 | Biparjoy | ARB | 2023 | 97 |
| 2023129N08091 | Mocha | BOB | 2023 | 40 |
| 2019296N15066 | Kyarr | ARB | 2019 | 74 |
| ... and 14 more | | | | |

## Model

**HistGradientBoostingRegressor** (scikit-learn) — a gradient-boosted decision
tree ensemble. One model per target variable × lead time = 12 models total.

- **Targets**: lat, lon, wind_kt, pressure_hpa
- **Lead times**: +24h, +48h, +72h
- **Features**: current position, intensity, basin encoding, time features,
  recent motion vector (delta-lat/lon over 1 and 2 steps), intensity change rate
- **Split**: Storm-level train/test (no timestamp leakage)

## Quick Start

```bash
# 1. Generate storms.csv from IBTrACS
python -m src.data_pipeline

# 2. Train model + split data
python -m src.model

# 3. Evaluate on held-out storms
python -m src.evaluate

# 4. Run tests
python -m pytest tests/ -v
```

## File Structure (this branch)

```
data/
  raw/ibtracs.NI.list.v04r01.csv    # Original IBTrACS download
  processed/storms.csv               # Cleaned, CONTRACT.md schema
  processed/train_storm_ids.json     # Train split
  processed/test_storm_ids.json      # Test split
src/
  data_pipeline.py                   # IBTrACS -> storms.csv
  model.py                          # predict_track_intensity()
  evaluate.py                       # Evaluation + metrics
models/
  gb_models.joblib                   # Trained model artifacts
results/
  eval_metrics.json                  # Per-lead-time metrics
tests/
  test_split.py                     # Split integrity + contract tests
RESULTS.md                          # Honest evaluation summary
CONTRACT.md                         # Shared team contract (frozen)
```

## CONTRACT.md Compliance

- `predict_track_intensity(storm_id, lead_times_hours)` returns exactly:
  ```json
  {
    "storm_id": "...",
    "track": [{"lead_h": 24, "lat": ..., "lon": ...}, ...],
    "intensity": [{"lead_h": 24, "wind_kt": ..., "pressure_hpa": ...}, ...]
  }
  ```
- Falls back to MOCK predictions (realistic fake numbers in correct shape) if
  the trained model isn't available, so downstream teammates are never blocked.

## What's NOT in This Sprint

Explicitly out of scope per CONTRACT.md:
- Satellite imagery ingestion
- CNNs / deep learning
- Dvorak classification
- Grad-CAM explanations
- Real SMS alerts

## Dependencies

- Python 3.10+
- numpy, pandas, scikit-learn, joblib, pytest
- See `requirements.txt`
