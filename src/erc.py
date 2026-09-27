"""
erc.py -- ChakraNetra Eyewall Replacement Cycle Detection Module

Implements three ERC components per Master Plan v3 SS5.7-SS5.9:
  Component A: Direct IR classifier (heuristic proxy for CNN/ViT + INSAT)
  Component B: Synthetic PMW (gated -- returns null)
  Component C: RMW-jump validator (post-hoc -- returns null in real-time)

ERC is the single strongest novelty claim in the project: the first
automated ERC detector for the North Indian Ocean basin, and the first
built entirely on geostationary IR without passive microwave dependency.

CURRENT STATUS: Physics-based pattern detection using IBTrACS intensity
history. Follows the exact v3 API contract (Appendix B).

Key physical insight (SS2, v3): During ERC, peak wind drops for ~16h on
average while the radius of destructive winds EXPANDS 30-60 km -- the
opposite of what a falling wind number suggests. This module exists to
catch that counter-intuitive case.
"""

import numpy as np
import pandas as pd

# ERC thresholds
ERC_WATCH_THRESHOLD = 0.6   # P(ERC) > this triggers ERC WATCH overlay
ERC_MIN_WIND_KT = 96        # Cat 3+ required for ERC
ERC_TYPICAL_DURATION_H = 16  # Average ERC duration (hours)

# Outer/inner eyewall RMW expansion factor during ERC
# During ERC, outer eyewall is typically 1.5-2.5x the inner RMW
ERC_RMW_EXPANSION = 1.8


# --------------------------------------------------------------------------- #
# Intensity pattern detection
# --------------------------------------------------------------------------- #

def _detect_intensity_oscillation(history_df: pd.DataFrame) -> dict:
    """
    Detect the characteristic ERC intensity signature:
    Peak -> Dip (wind drops 10-30 kt) -> Reintensification.

    During ERC:
    - Inner eyewall contracts and weakens
    - Outer eyewall forms and contracts inward
    - Peak wind drops but destructive wind RADIUS expands
    - Storm typically reintensifies past original peak
    """
    if len(history_df) < 5:
        return {
            "has_oscillation": False, "phase": "insufficient_data",
            "peak_wind": 0.0, "current_wind": 0.0, "dip_magnitude": 0.0,
            "steps_since_peak": 0,
        }

    winds = history_df["wind_kt"].values
    current = float(winds[-1])

    # Find the most recent local maximum
    peak_idx = -1
    peak_wind = 0.0
    for i in range(len(winds) - 2, 0, -1):
        if winds[i] >= winds[i - 1] and winds[i] >= winds[i + 1]:
            if winds[i] > peak_wind:
                peak_wind = float(winds[i])
                peak_idx = i
                break

    if peak_idx < 0 or peak_wind < ERC_MIN_WIND_KT:
        return {
            "has_oscillation": False, "phase": "no_qualifying_peak",
            "peak_wind": round(peak_wind, 1), "current_wind": round(current, 1),
            "dip_magnitude": 0.0, "steps_since_peak": 0,
        }

    dip_magnitude = peak_wind - current
    steps_since_peak = len(winds) - 1 - peak_idx

    # ERC dip characteristics:
    # - 10-35 kt drop from peak (not a full weakening/landfall collapse)
    # - Occurs within 2-6 timesteps of peak (~12-36 hours)
    # - Storm is still strong (>64 kt)
    if (10 <= dip_magnitude <= 35
            and 1 <= steps_since_peak <= 6
            and current > 64):
        phase = "active_dip"
        has_oscillation = True
    elif (dip_magnitude > 5
          and steps_since_peak <= 3
          and peak_wind >= ERC_MIN_WIND_KT
          and current > 80):
        phase = "possible_onset"
        has_oscillation = True
    elif (dip_magnitude < 5
          and peak_wind >= ERC_MIN_WIND_KT
          and steps_since_peak <= 2):
        phase = "pre_erc"
        has_oscillation = False
    else:
        phase = "no_erc_pattern"
        has_oscillation = False

    return {
        "has_oscillation": has_oscillation,
        "phase": phase,
        "peak_wind": round(peak_wind, 1),
        "current_wind": round(current, 1),
        "dip_magnitude": round(dip_magnitude, 1),
        "steps_since_peak": steps_since_peak,
    }


# --------------------------------------------------------------------------- #
# Component A: Direct IR classifier
# --------------------------------------------------------------------------- #

def erc_direct(history_df: pd.DataFrame, current_wind: float,
               basin: str) -> dict:
    """
    Component A: Direct IR classifier -- proxy for CNN/ViT + INSAT.

    In production: CNN/ViT encoder on 3-channel INSAT stack (TIR1,
    TIR1-TIR2 split-window, WV) fused with tabular features (shear,
    SST/OHC proxy, current Vmax, RMW trend).

    Prototype: Pattern-based detection using intensity history.

    Returns:
        {"direct_probability": float, "phase": str, "details": dict}
    """
    oscillation = _detect_intensity_oscillation(history_df)

    if not oscillation["has_oscillation"]:
        base_prob = 0.0
        # Minor elevation if storm is very strong (Cat 4/5)
        # because ERC is more common in intense storms
        if current_wind >= 130:
            base_prob = 0.15
        elif current_wind >= 113:
            base_prob = 0.08
        elif current_wind >= ERC_MIN_WIND_KT:
            base_prob = 0.05
    else:
        phase = oscillation["phase"]
        dip = oscillation["dip_magnitude"]

        if phase == "active_dip":
            # Classic ERC signature -- high probability
            base_prob = 0.55 + min(0.35, dip / 60.0)
            if oscillation["peak_wind"] >= 130:
                base_prob += 0.1  # More common in very intense storms
        elif phase == "possible_onset":
            base_prob = 0.35 + min(0.2, dip / 40.0)
        else:
            base_prob = 0.1

    probability = max(0.0, min(0.95, base_prob))

    return {
        "direct_probability": round(probability, 3),
        "phase": oscillation["phase"],
        "details": {
            "peak_wind_kt": oscillation["peak_wind"],
            "current_wind_kt": oscillation["current_wind"],
            "dip_magnitude_kt": oscillation["dip_magnitude"],
            "steps_since_peak": oscillation.get("steps_since_peak", 0),
        },
    }


# --------------------------------------------------------------------------- #
# Component B: Synthetic PMW (GATED)
# --------------------------------------------------------------------------- #

def erc_synthetic() -> dict:
    """
    Component B: Synthetic PMW -- GATED (not built).

    Would use a diffusion transformer (Li, Tan & Bai 2026) to synthesize
    PMW-equivalent 85-92 GHz brightness temperatures from IR input,
    then compute an ARCHER-style ring score.

    Returns null per v3 contract -- go/no-go checkpoint (Week 5, Day 32)
    not yet reached.
    """
    return {
        "synthetic_probability": None,
        "status": "gated_pending_go_nogo",
    }


# --------------------------------------------------------------------------- #
# Component C: RMW-jump validator (POST-HOC)
# --------------------------------------------------------------------------- #

def erc_rmw_check() -> dict:
    """
    Component C: RMW-Jump validator -- POST-HOC only.

    Checks IBTrACS for an RMW jump 12-36h after a flagged ERC event,
    following Fosler-Lussier & Wang (2025).  Runs after the fact on the
    validation set, not in real-time.

    Returns null per v3 contract -- requires updated best-track data.
    """
    return {
        "rmw_jump_confirmed": None,
        "status": "post_hoc_not_realtime",
    }


# --------------------------------------------------------------------------- #
# Combined ERC assessment
# --------------------------------------------------------------------------- #

def compute_erc(history_df: pd.DataFrame, current_wind: float,
                basin: str) -> dict:
    """
    Compute full ERC assessment with all three components.

    Per v3 API contract (Appendix B):
    {
        "direct_probability": float,
        "synthetic_probability": null,
        "rmw_jump_confirmed": null,
        "reference_rmw_source": "inner" | "outer"
    }

    When direct_probability >= ERC_WATCH_THRESHOLD, switches
    reference_rmw_source to "outer" so risk.gis uses the expanding
    outer eyewall for wind radii (CN-014).
    """
    direct = erc_direct(history_df, current_wind, basin)
    synthetic = erc_synthetic()
    rmw_check = erc_rmw_check()

    # Reference RMW source switches to outer when ERC is likely (CN-014)
    direct_prob = direct["direct_probability"]
    reference_rmw_source = "outer" if direct_prob >= ERC_WATCH_THRESHOLD else "inner"

    return {
        "direct_probability": direct["direct_probability"],
        "synthetic_probability": synthetic["synthetic_probability"],
        "rmw_jump_confirmed": rmw_check["rmw_jump_confirmed"],
        "reference_rmw_source": reference_rmw_source,
        "phase": direct.get("phase", "unknown"),
        "details": direct.get("details", {}),
    }
