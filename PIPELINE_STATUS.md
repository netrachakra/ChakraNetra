# PIPELINE_STATUS.md -- What's Real vs. What's Mocked

> **For the presenting team**: this is what you can honestly claim on stage.

## TL;DR

**Everything in the demo is running against real data and real trained models.**
No mocks are active. The MOCK fallback code exists in model.py and api.py for
graceful degradation, but it is never triggered during normal operation.

## Storm-by-Storm Integration Results

All 5 tested storms passed the full pipeline with zero errors:

| Storm ID | Pipeline | Prediction | Calibration | Risk | API | Status |
|----------|----------|-----------|-------------|------|-----|--------|
| 2019160N11073 | REAL | REAL | REAL | REAL | REAL | PASS |
| 2019301N05081 | REAL | REAL | REAL | REAL | REAL | PASS |
| 2018141N08059 | REAL | REAL | REAL | REAL | REAL | PASS |
| 2018279N11069 | REAL | REAL | REAL | REAL | REAL | PASS |
| 2018281N14088 | REAL | REAL | REAL | REAL | REAL | PASS |

All 20 storms in the dataset work (tested via API integration tests).

## Module-by-Module Status

### Data (src/data_pipeline.py) -- REAL
- Source: NOAA IBTrACS v04r01, North Indian Ocean basin
- 20 real cyclones, 993 observations, 2018-2023
- Mix of Bay of Bengal (BOB) and Arabian Sea (ARB) storms
- Includes Fani (2019), Amphan (2020), Tauktae (2021), Mocha (2023), Biparjoy (2023)
- Storm IDs are IBTrACS SIDs (e.g., `2019116N02090`), NOT the human-readable
  format in CONTRACT.md examples (`BOB-2023-04`). This is a known simplification.

### Model (src/model.py) -- REAL
- HistGradientBoostingRegressor (scikit-learn), not LSTM or dynamical model
- 12 models: 4 targets (lat, lon, wind_kt, pressure_hpa) x 3 lead times (24/48/72h)
- Features: position, motion vector (1-step and 2-step), intensity change, basin, time
- Trained on 15 storms, evaluated on 5 held-out storms
- Track error: 280-736 km (+24h to +72h) -- honest baseline, not competitive with NWP
- MOCK fallback exists but is only triggered if gb_models.joblib is missing

### Calibration (src/calibration.py) -- REAL
- Split-conformal prediction on 5 held-out test storms
- Conformal quantiles: +24h 318km/30kt, +48h 685km/31kt, +72h 860km/36kt
- Empirical coverage: **0.807** (measured, not hardcoded)
- If conformal_scores.json is missing, falls back to conservative fixed values
  with empirical_coverage = 0.0 (signaling uncalibrated)

### Risk (src/risk.py) -- REAL
- Modified Rankine vortex wind field (exponent 0.5 outside RMW)
- RMW estimated via linear regression: `RMW_km = max(15, 46.4 - 0.22 * Vmax_kt)`
  - This is a statistical approximation, NOT from recon or SAR data
- Risk score: monotonic piecewise function mapping wind to [0, 1]
- Wind radii at 34/50/64 kt thresholds from analytical Rankine inversion
- Population-density weighting: NOT implemented (out of scope)

### API (src/api.py) -- REAL
- FastAPI, all three real modules wired in
- POST /v1/predict merges predict + calibrate + risk
- GET /v1/health, GET /v1/storms
- Pydantic validation with clean 404/422 errors
- MOCK fallbacks exist but are never triggered (all modules import successfully)

### Dashboard (dashboard/app.py) -- REAL
- Streamlit + Folium
- Shows actual IBTrACS track + predicted track with calibrated cone + wind radii
- API or direct-import fallback via sidebar checkbox
- Lat/lon verified against Cyclone Fani (3.2N, 89.8E = Bay of Bengal)

## What You CAN Claim on Stage

1. "We trained on real NOAA IBTrACS best-track data, not synthetic data"
2. "Our uncertainty intervals have measured 80.7% coverage, verified on held-out storms"
3. "The wind risk model uses a physics-based Rankine vortex, not arbitrary numbers"
4. "The full pipeline runs in under 150ms per prediction (after cold start)"
5. "All 51 automated tests pass on the integrated codebase"

## What You Should NOT Claim

1. "Our model is competitive with IMD/JTWC/NWP forecasting" -- it's not, 280km+ track error
2. "We use satellite imagery / CNN / deep learning" -- we don't, it's gradient-boosted trees
3. "We have population-weighted risk scoring" -- risk_score is wind-only, no population data
4. "We send real SMS/WhatsApp alerts" -- no alert system built
5. "The RMW is observed data" -- it's a statistical estimate from a linear regression

## What's Explicitly Out of Scope (Master Plan items)

- Satellite imagery ingestion
- CNN / deep learning models
- Dvorak classification
- Grad-CAM explainability
- Real SMS/WhatsApp alerts
- Population-density-weighted risk
- GIS map rendering (we use Folium, not a full GIS)
- Authentication / rate limiting on the API

## Tests

51 tests pass on main:
- test_api.py: 20 (API integration + error handling)
- test_calibration.py: 9 (calibration contract + coverage check)
- test_risk.py: 17 (monotonicity + Rankine + radii + contract)
- test_split.py: 5 (split integrity + data schema)
