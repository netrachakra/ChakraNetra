"""
risk.py -- ChakraNetra wind-risk assessment

Implements compute_risk() and rankine_vortex() per CONTRACT.md Function Contract 3.

Wind field model: Modified Rankine vortex
  Inside  RMW: V(r) = Vmax * (r / RMW)
  Outside RMW: V(r) = Vmax * (RMW / r) ** 0.5

RMW estimation: Simple regression against max sustained wind speed.
  Based on empirical fits from Willoughby & Rahn (2004) and Knaff & Zehr (2007)
  for North Indian Ocean cyclones. This is a rough approximation; real systems
  use recon data or satellite-based estimates.

  Regression used (NI basin approximation):
      RMW_km = max(15, 46.4 - 0.22 * Vmax_kt)
  This yields ~40 km for weak storms and ~15 km for intense ones, consistent
  with observed NI basin cyclone eye sizes.

  ASSUMPTION: This RMW estimate is a statistical approximation only. It does
  not account for storm size variability, asymmetry, latitude dependence, or
  outer wind structure. For a production system, use observed RMW from recon
  or SAR data.
"""

import math


# --------------------------------------------------------------------------- #
# RMW estimation
# --------------------------------------------------------------------------- #

def estimate_rmw_km(vmax_kt: float) -> float:
    """
    Estimate radius of maximum wind (RMW) from max sustained wind speed.

    Regression: RMW_km = max(15, 46.4 - 0.22 * Vmax_kt)

    ASSUMPTION: Simple linear regression approximation for NI basin.
    Based on empirical fits from Willoughby & Rahn (2004) for Atlantic
    storms, adapted with a smaller intercept for the typically compact
    NI basin cyclones. Real RMW varies significantly by storm and should
    ideally come from aircraft reconnaissance or SAR satellite data.

    Args:
        vmax_kt: Maximum sustained wind speed in knots.

    Returns:
        Estimated RMW in kilometers.
    """
    # Linear regression: smaller storms (higher Vmax) have smaller eyes
    rmw = 46.4 - 0.22 * vmax_kt
    # Floor at 15 km (intense storms can't have arbitrarily small RMW)
    return max(15.0, rmw)


# --------------------------------------------------------------------------- #
# Modified Rankine vortex wind profile
# --------------------------------------------------------------------------- #

def rankine_vortex(r_km: float, vmax_kt: float, rmw_km: float = None) -> float:
    """
    Modified Rankine vortex wind speed at radius r.

    Inside  RMW: V(r) = Vmax * (r / RMW)       -- linear ramp
    Outside RMW: V(r) = Vmax * (RMW / r) ** 0.5 -- modified decay (exponent 0.5)

    The exponent 0.5 (instead of the classic 1.0) gives a broader wind field
    that better matches observed tropical cyclone profiles, especially in the
    outer regions.

    Args:
        r_km:    Radius from storm center in km.
        vmax_kt: Maximum sustained wind in knots.
        rmw_km:  Radius of maximum wind in km. If None, estimated from vmax_kt.

    Returns:
        Wind speed in knots at radius r.
    """
    if rmw_km is None:
        rmw_km = estimate_rmw_km(vmax_kt)

    if r_km <= 0:
        return 0.0

    if rmw_km <= 0:
        return 0.0

    if r_km <= rmw_km:
        # Inside RMW: linear increase
        return vmax_kt * (r_km / rmw_km)
    else:
        # Outside RMW: modified decay with exponent 0.5
        return vmax_kt * (rmw_km / r_km) ** 0.5


# --------------------------------------------------------------------------- #
# Wind radii computation
# --------------------------------------------------------------------------- #

def _find_wind_radius_km(
    threshold_kt: float, vmax_kt: float, rmw_km: float
) -> float:
    """
    Find the radius at which the Rankine vortex wind drops to threshold_kt.

    For the outer region: V = Vmax * (RMW / r) ^ 0.5
    Solving for r: r = RMW * (Vmax / threshold) ^ 2

    If Vmax < threshold, the storm never reaches that wind speed, return 0.
    """
    if vmax_kt < threshold_kt:
        return 0.0

    # Outer profile: V = Vmax * (RMW/r)^0.5
    # threshold = Vmax * (RMW/r)^0.5
    # (RMW/r)^0.5 = threshold / Vmax
    # RMW/r = (threshold / Vmax)^2
    # r = RMW / (threshold / Vmax)^2
    # r = RMW * (Vmax / threshold)^2
    r = rmw_km * (vmax_kt / threshold_kt) ** 2
    return round(r, 1)


def compute_wind_radii(vmax_kt: float, rmw_km: float = None) -> dict:
    """
    Compute the radii at which wind drops to 34, 50, and 64 kt thresholds.

    Args:
        vmax_kt: Maximum sustained wind in knots.
        rmw_km:  Radius of maximum wind. If None, estimated.

    Returns:
        {"34kt": float, "50kt": float, "64kt": float} -- radii in km.
    """
    if rmw_km is None:
        rmw_km = estimate_rmw_km(vmax_kt)

    return {
        "34kt": _find_wind_radius_km(34.0, vmax_kt, rmw_km),
        "50kt": _find_wind_radius_km(50.0, vmax_kt, rmw_km),
        "64kt": _find_wind_radius_km(64.0, vmax_kt, rmw_km),
    }


# --------------------------------------------------------------------------- #
# Risk score
# --------------------------------------------------------------------------- #

def _wind_risk_score(vmax_kt: float) -> float:
    """
    Compute a normalized risk score from wind speed.

    Uses a piecewise function inspired by the Saffir-Simpson scale:
      - Below 34 kt (tropical depression): low risk, score 0.0 - 0.2
      - 34-63 kt (tropical storm):         moderate,  score 0.2 - 0.4
      - 64-82 kt (Cat 1):                  high,      score 0.4 - 0.55
      - 83-95 kt (Cat 2):                  very high, score 0.55 - 0.7
      - 96-112 kt (Cat 3):                 severe,    score 0.7 - 0.85
      - 113-136 kt (Cat 4):                extreme,   score 0.85 - 0.95
      - >136 kt (Cat 5):                   catastrophic, score 0.95 - 1.0

    This is a monotonically increasing function: stronger wind = higher score.
    """
    if vmax_kt <= 0:
        return 0.0

    # Thresholds and corresponding score ranges
    bands = [
        (34,  0.0,  0.2),
        (64,  0.2,  0.4),
        (83,  0.4,  0.55),
        (96,  0.55, 0.7),
        (113, 0.7,  0.85),
        (137, 0.85, 0.95),
    ]

    prev_threshold = 0.0
    for upper, score_low, score_high in bands:
        if vmax_kt <= upper:
            frac = (vmax_kt - prev_threshold) / (upper - prev_threshold)
            return round(score_low + frac * (score_high - score_low), 4)
        prev_threshold = upper

    # Above Cat 5 threshold (137+ kt)
    # Asymptotically approach 1.0
    excess = vmax_kt - 137.0
    return round(min(1.0, 0.95 + 0.05 * (1 - math.exp(-excess / 30.0))), 4)


# --------------------------------------------------------------------------- #
# compute_risk() -- CONTRACT.md Function Contract 3
# --------------------------------------------------------------------------- #

def compute_risk(intensity_kt: float, lat: float, lon: float) -> dict:
    """
    Compute wind-based risk score and wind radii for a cyclone.

    Per CONTRACT.md:
        {
            "risk_score": float,     # 0.0 to 1.0
            "wind_radii_km": {
                "34kt": float,
                "50kt": float,
                "64kt": float,
            }
        }

    Args:
        intensity_kt: Maximum sustained wind speed in knots.
        lat: Latitude of the storm center (currently unused; reserved
             for future population-density weighting).
        lon: Longitude of the storm center (currently unused; reserved
             for future population-density weighting).

    Returns:
        Risk assessment dict per CONTRACT.md.
    """
    # lat, lon are accepted per contract but not used for risk weighting
    # in this sprint. Population-density weighting is out of scope.
    _ = lat, lon

    rmw = estimate_rmw_km(intensity_kt)
    radii = compute_wind_radii(intensity_kt, rmw)
    score = _wind_risk_score(intensity_kt)

    return {
        "risk_score": score,
        "wind_radii_km": radii,
    }


# --------------------------------------------------------------------------- #
# CLI demo
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("=== ChakraNetra Risk Assessment ===\n")

    test_cases = [
        (30, 15.0, 85.0, "Tropical Depression"),
        (50, 16.0, 86.0, "Tropical Storm"),
        (75, 18.0, 88.0, "Category 1"),
        (90, 19.0, 87.0, "Category 2"),
        (105, 20.0, 86.0, "Category 3"),
        (125, 21.0, 85.0, "Category 4"),
        (145, 22.0, 84.0, "Category 5"),
    ]

    for wind, lat, lon, label in test_cases:
        result = compute_risk(wind, lat, lon)
        rmw = estimate_rmw_km(wind)
        print(f"{label:22s} ({wind:3d} kt): "
              f"risk={result['risk_score']:.3f}, "
              f"RMW={rmw:.0f}km, "
              f"R34={result['wind_radii_km']['34kt']:.0f}km, "
              f"R50={result['wind_radii_km']['50kt']:.0f}km, "
              f"R64={result['wind_radii_km']['64kt']:.0f}km")

    print("\n--- Rankine vortex profile for 100kt storm ---")
    rmw = estimate_rmw_km(100)
    for r in [0, 5, 10, rmw, rmw*1.5, rmw*2, rmw*3, rmw*5, rmw*10]:
        v = rankine_vortex(r, 100, rmw)
        print(f"  r={r:6.1f} km -> V={v:5.1f} kt")
