"""
api.py -- ChakraNetra FastAPI backend

Exposes POST /v1/predict per CONTRACT.md API contract, plus
GET /v1/health and GET /v1/storms utility endpoints.

Wires together predict_track_intensity(), calibrate(), and compute_risk()
from the real modules. Falls back to inline MOCK if imports fail, so the
API can always start standalone.

No auth, no rate limiting, no database -- all out of scope for a 3-day demo.
"""

import os
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------- #
# Try importing real modules; fall back to mocks if unavailable
# --------------------------------------------------------------------------- #
_USING_REAL_MODEL = False
_USING_REAL_CALIBRATION = False
_USING_REAL_RISK = False

try:
    from src.model import predict_track_intensity
    _USING_REAL_MODEL = True
except Exception:
    predict_track_intensity = None  # type: ignore

try:
    from src.calibration import calibrate
    _USING_REAL_CALIBRATION = True
except Exception:
    calibrate = None  # type: ignore

try:
    from src.risk import compute_risk
    _USING_REAL_RISK = True
except Exception:
    compute_risk = None  # type: ignore

# --------------------------------------------------------------------------- #
# Storms CSV for listing available storm_ids
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
STORMS_CSV = BASE_DIR / "data" / "processed" / "storms.csv"

_STORM_IDS: list[str] = []


def _load_storm_ids() -> list[str]:
    """Load available storm IDs from storms.csv."""
    global _STORM_IDS
    if _STORM_IDS:
        return _STORM_IDS
    if STORMS_CSV.exists():
        import pandas as pd
        df = pd.read_csv(STORMS_CSV, usecols=["storm_id"])
        _STORM_IDS = sorted(df["storm_id"].unique().tolist())
    return _STORM_IDS


# --------------------------------------------------------------------------- #
# MOCK fallbacks (exact CONTRACT.md shape, clearly fake numbers)
# --------------------------------------------------------------------------- #

def _mock_predict_track_intensity(storm_id: str, lead_times_hours: list[int]) -> dict:
    """MOCK: returns plausible fake numbers in CONTRACT.md shape."""
    return {
        "storm_id": storm_id,
        "track": [
            {"lead_h": int(h), "lat": round(15.0 + h * 0.03, 2),
             "lon": round(85.0 - h * 0.01, 2)}
            for h in lead_times_hours
        ],
        "intensity": [
            {"lead_h": int(h), "wind_kt": round(65.0 - h * 0.1, 1),
             "pressure_hpa": round(985.0 + h * 0.08, 1)}
            for h in lead_times_hours
        ],
    }


def _mock_calibrate(raw_prediction: dict) -> dict:
    """MOCK: adds calibration keys with placeholder values."""
    result = {
        "storm_id": raw_prediction["storm_id"],
        "track": [],
        "intensity": [],
        "empirical_coverage": 0.0,  # signals uncalibrated
    }
    for pt in raw_prediction["track"]:
        result["track"].append({
            **pt,
            "cone_km_lower": 0.0,
            "cone_km_upper": 300.0 + pt["lead_h"] * 5.0,
        })
    for pt in raw_prediction["intensity"]:
        result["intensity"].append({
            **pt,
            "interval_kt": [
                round(max(0, pt["wind_kt"] - 20), 1),
                round(pt["wind_kt"] + 20, 1),
            ],
        })
    return result


def _mock_compute_risk(intensity_kt: float, lat: float, lon: float) -> dict:
    """MOCK: returns placeholder risk dict."""
    score = min(1.0, intensity_kt / 150.0)
    return {
        "risk_score": round(score, 4),
        "wind_radii_km": {"34kt": 150.0, "50kt": 80.0, "64kt": 40.0},
    }


# --------------------------------------------------------------------------- #
# Resolve which implementation to use
# --------------------------------------------------------------------------- #
_predict_fn = predict_track_intensity if _USING_REAL_MODEL else _mock_predict_track_intensity
_calibrate_fn = calibrate if _USING_REAL_CALIBRATION else _mock_calibrate
_risk_fn = compute_risk if _USING_REAL_RISK else _mock_compute_risk


# --------------------------------------------------------------------------- #
# Pydantic models for request/response validation
# --------------------------------------------------------------------------- #

class PredictRequest(BaseModel):
    """POST /v1/predict request body."""
    storm_id: str = Field(..., min_length=1, description="Storm identifier from storms.csv")
    lead_times_hours: list[int] = Field(
        default=[24, 48, 72],
        description="Forecast lead times in hours",
    )

    @field_validator("lead_times_hours")
    @classmethod
    def validate_lead_times(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("lead_times_hours must not be empty")
        for h in v:
            if h <= 0:
                raise ValueError(f"Lead time must be positive, got {h}")
            if h > 240:
                raise ValueError(f"Lead time {h}h exceeds maximum 240h")
        return v


class TrackPoint(BaseModel):
    lead_h: int
    lat: float
    lon: float
    cone_km_lower: float | None = None
    cone_km_upper: float | None = None


class IntensityPoint(BaseModel):
    lead_h: int
    wind_kt: float
    pressure_hpa: float
    interval_kt: list[float] | None = None


class RiskResult(BaseModel):
    risk_score: float
    wind_radii_km: dict[str, float]


class PredictResponse(BaseModel):
    """POST /v1/predict response -- merges all three contracts + model_version."""
    storm_id: str
    track: list[TrackPoint]
    intensity: list[IntensityPoint]
    empirical_coverage: float | None = None
    risk: RiskResult | None = None
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    calibration_loaded: bool
    risk_loaded: bool


class StormsResponse(BaseModel):
    storm_ids: list[str]
    count: int


class ErrorResponse(BaseModel):
    detail: str


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #

MODEL_VERSION = "chakranetra-v0.1.0-gbr"

app = FastAPI(
    title="ChakraNetra API",
    description="AI Cyclone Track + Intensity Forecasting -- SIH Hackathon Prototype",
    version=MODEL_VERSION,
)

# CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/v1/health", response_model=HealthResponse)
def health():
    """Liveness check. Returns which real modules are loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=_USING_REAL_MODEL,
        calibration_loaded=_USING_REAL_CALIBRATION,
        risk_loaded=_USING_REAL_RISK,
    )


@app.get("/v1/storms", response_model=StormsResponse)
def list_storms():
    """List all storm_ids available in data/processed/storms.csv."""
    ids = _load_storm_ids()
    return StormsResponse(storm_ids=ids, count=len(ids))


@app.post(
    "/v1/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        404: {"model": ErrorResponse, "description": "Storm not found"},
    },
)
def predict(request: PredictRequest):
    """
    Predict cyclone track + intensity at given lead times.

    Merges predict_track_intensity(), calibrate(), and compute_risk()
    into one response per CONTRACT.md API contract.
    """
    storm_ids = _load_storm_ids()

    # Validate storm_id exists (if we have data loaded)
    if storm_ids and request.storm_id not in storm_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Storm '{request.storm_id}' not found. "
                   f"Available: {storm_ids[:5]}{'...' if len(storm_ids) > 5 else ''}",
        )

    # Step 1: predict_track_intensity
    try:
        raw_prediction = _predict_fn(request.storm_id, request.lead_times_hours)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {e}",
        )

    # Step 2: calibrate
    try:
        calibrated = _calibrate_fn(raw_prediction)
    except Exception as e:
        # If calibration fails, proceed without it
        calibrated = raw_prediction
        calibrated["empirical_coverage"] = None

    # Step 3: compute_risk (using the first/strongest intensity point)
    risk_result = None
    try:
        if calibrated.get("intensity"):
            # Use the strongest predicted wind for risk assessment
            strongest = max(calibrated["intensity"], key=lambda p: p["wind_kt"])
            # Get lat/lon from corresponding track point
            matching_track = next(
                (t for t in calibrated["track"] if t["lead_h"] == strongest["lead_h"]),
                calibrated["track"][0] if calibrated["track"] else None,
            )
            if matching_track:
                risk_result = _risk_fn(
                    strongest["wind_kt"],
                    matching_track["lat"],
                    matching_track["lon"],
                )
    except Exception:
        risk_result = None

    # Assemble response
    return PredictResponse(
        storm_id=calibrated["storm_id"],
        track=[TrackPoint(**pt) for pt in calibrated["track"]],
        intensity=[IntensityPoint(**pt) for pt in calibrated["intensity"]],
        empirical_coverage=calibrated.get("empirical_coverage"),
        risk=RiskResult(**risk_result) if risk_result else None,
        model_version=MODEL_VERSION,
    )


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
