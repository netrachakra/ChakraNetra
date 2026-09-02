"""
app.py -- ChakraNetra Streamlit Dashboard

Single-screen cyclone forecasting demo:
  - Dropdown to select a storm
  - Folium map: actual track + predicted track with calibrated cone + wind radii
  - Sidebar table: wind intensity with calibrated intervals per lead time

Data source: API at http://localhost:8000 if reachable, else direct imports.
"""

import os
import sys

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# --------------------------------------------------------------------------- #
# Ensure project root is on path for direct-import fallback
# --------------------------------------------------------------------------- #
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

STORMS_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "storms.csv")
API_BASE = os.environ.get("CHAKRANETRA_API_URL", "http://localhost:8000")


# --------------------------------------------------------------------------- #
# Backend communication (API or direct fallback)
# --------------------------------------------------------------------------- #

def _api_available(base_url: str) -> bool:
    """Check if the FastAPI backend is reachable."""
    try:
        import httpx
        r = httpx.get(f"{base_url}/v1/health", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def _get_storm_ids_api(base_url: str) -> list[str]:
    import httpx
    r = httpx.get(f"{base_url}/v1/storms", timeout=5.0)
    return r.json()["storm_ids"]


def _predict_api(base_url: str, storm_id: str, lead_times: list[int]) -> dict:
    import httpx
    r = httpx.post(
        f"{base_url}/v1/predict",
        json={"storm_id": storm_id, "lead_times_hours": lead_times},
        timeout=15.0,
    )
    r.raise_for_status()
    return r.json()


def _get_storm_ids_direct() -> list[str]:
    df = pd.read_csv(STORMS_CSV, usecols=["storm_id"])
    return sorted(df["storm_id"].unique().tolist())


def _predict_direct(storm_id: str, lead_times: list[int]) -> dict:
    """Chain predict -> calibrate -> risk directly."""
    from src.model import predict_track_intensity
    from src.calibration import calibrate
    from src.risk import compute_risk

    raw = predict_track_intensity(storm_id, lead_times)
    cal = calibrate(raw)

    # Compute risk from strongest predicted wind
    risk_result = None
    if cal.get("intensity"):
        strongest = max(cal["intensity"], key=lambda p: p["wind_kt"])
        matching_track = next(
            (t for t in cal["track"] if t["lead_h"] == strongest["lead_h"]),
            cal["track"][0] if cal["track"] else None,
        )
        if matching_track:
            risk_result = compute_risk(
                strongest["wind_kt"],
                matching_track["lat"],
                matching_track["lon"],
            )

    result = {
        "storm_id": cal["storm_id"],
        "track": cal["track"],
        "intensity": cal["intensity"],
        "empirical_coverage": cal.get("empirical_coverage"),
        "risk": risk_result,
        "model_version": "direct-import",
    }
    return result


# --------------------------------------------------------------------------- #
# Map building
# --------------------------------------------------------------------------- #

# Color scheme
ACTUAL_TRACK_COLOR = "#2563EB"     # blue
PREDICTED_TRACK_COLOR = "#DC2626"  # red
CONE_COLOR = "#FCA5A5"            # light red
WIND_34_COLOR = "#FDE68A"         # yellow
WIND_50_COLOR = "#FDBA74"         # orange
WIND_64_COLOR = "#F87171"         # red


def _intensity_to_color(wind_kt: float) -> str:
    """Map wind speed to marker color (Saffir-Simpson-ish)."""
    if wind_kt >= 137:
        return "#7C3AED"   # purple - Cat 5
    elif wind_kt >= 113:
        return "#DC2626"   # red - Cat 4
    elif wind_kt >= 96:
        return "#EA580C"   # orange - Cat 3
    elif wind_kt >= 83:
        return "#D97706"   # amber - Cat 2
    elif wind_kt >= 64:
        return "#CA8A04"   # yellow - Cat 1
    elif wind_kt >= 34:
        return "#059669"   # green - TS
    else:
        return "#6B7280"   # gray - TD


def build_map(
    storm_df: pd.DataFrame,
    prediction: dict,
) -> folium.Map:
    """
    Build a Folium map with:
    1. Actual historical track (blue polyline + circle markers)
    2. Predicted track (red dashed polyline + triangle markers)
    3. Calibrated cone overlay (translucent red circles at each predicted point)
    4. Wind radii circles (34/50/64 kt) at the strongest predicted point
    """
    # Center map on the storm's midpoint
    center_lat = storm_df["lat"].mean()
    center_lon = storm_df["lon"].mean()
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles="CartoDB positron",
    )

    # --- 1. Actual historical track ---
    actual_coords = list(zip(storm_df["lat"], storm_df["lon"]))
    folium.PolyLine(
        actual_coords,
        color=ACTUAL_TRACK_COLOR,
        weight=3,
        opacity=0.8,
        tooltip="Actual Track (IBTrACS)",
    ).add_to(m)

    # Mark each actual observation with intensity-colored dot
    for _, row in storm_df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            color=_intensity_to_color(row["wind_kt"]),
            fill=True,
            fill_opacity=0.8,
            tooltip=(
                f"Actual | {row['timestamp']}\n"
                f"Wind: {row['wind_kt']:.0f} kt | "
                f"Pres: {row['pressure_hpa']:.0f} hPa"
            ),
        ).add_to(m)

    # Mark genesis (first obs) with a star-like marker
    genesis = storm_df.iloc[0]
    folium.Marker(
        location=[genesis["lat"], genesis["lon"]],
        icon=folium.Icon(color="blue", icon="play", prefix="fa"),
        tooltip=f"Genesis: {genesis['timestamp']}",
    ).add_to(m)

    # --- 2. Predicted track ---
    if prediction and prediction.get("track"):
        # Get last actual position as start of prediction
        last_actual = storm_df.iloc[-1]
        pred_coords = [[last_actual["lat"], last_actual["lon"]]]

        for pt in prediction["track"]:
            pred_coords.append([pt["lat"], pt["lon"]])

            # Triangle marker for predicted positions
            folium.RegularPolygonMarker(
                location=[pt["lat"], pt["lon"]],
                number_of_sides=3,
                radius=8,
                color=PREDICTED_TRACK_COLOR,
                fill=True,
                fill_color=PREDICTED_TRACK_COLOR,
                fill_opacity=0.7,
                tooltip=f"Predicted +{pt['lead_h']}h | Lat: {pt['lat']}, Lon: {pt['lon']}",
            ).add_to(m)

            # --- 3. Calibrated cone overlay ---
            cone_upper = pt.get("cone_km_upper")
            if cone_upper and cone_upper > 0:
                folium.Circle(
                    location=[pt["lat"], pt["lon"]],
                    radius=cone_upper * 1000,  # km to meters
                    color=CONE_COLOR,
                    fill=True,
                    fill_color=CONE_COLOR,
                    fill_opacity=0.15,
                    weight=1,
                    tooltip=f"+{pt['lead_h']}h cone: {cone_upper:.0f} km radius",
                ).add_to(m)

        # Dashed predicted track line
        folium.PolyLine(
            pred_coords,
            color=PREDICTED_TRACK_COLOR,
            weight=2,
            dash_array="8 6",
            opacity=0.8,
            tooltip="Predicted Track",
        ).add_to(m)

    # --- 4. Wind radii circles ---
    risk = prediction.get("risk") if prediction else None
    if risk and risk.get("wind_radii_km"):
        # Place at strongest predicted point
        if prediction.get("intensity"):
            strongest = max(prediction["intensity"], key=lambda p: p["wind_kt"])
            matching = next(
                (t for t in prediction["track"] if t["lead_h"] == strongest["lead_h"]),
                None,
            )
            if matching:
                radii = risk["wind_radii_km"]
                for label, color, key in [
                    ("34 kt winds", WIND_34_COLOR, "34kt"),
                    ("50 kt winds", WIND_50_COLOR, "50kt"),
                    ("64 kt winds", WIND_64_COLOR, "64kt"),
                ]:
                    r_km = radii.get(key, 0)
                    if r_km > 0:
                        folium.Circle(
                            location=[matching["lat"], matching["lon"]],
                            radius=r_km * 1000,
                            color=color,
                            fill=True,
                            fill_color=color,
                            fill_opacity=0.12,
                            weight=2,
                            tooltip=f"{label}: {r_km:.0f} km radius",
                        ).add_to(m)

    # Fit map bounds to show everything
    all_lats = list(storm_df["lat"])
    all_lons = list(storm_df["lon"])
    if prediction and prediction.get("track"):
        all_lats += [pt["lat"] for pt in prediction["track"]]
        all_lons += [pt["lon"] for pt in prediction["track"]]
    if all_lats and all_lons:
        m.fit_bounds([
            [min(all_lats) - 1, min(all_lons) - 1],
            [max(all_lats) + 1, max(all_lons) + 1],
        ])

    return m


# --------------------------------------------------------------------------- #
# Streamlit app
# --------------------------------------------------------------------------- #

def main():
    st.set_page_config(
        page_title="ChakraNetra - Cyclone Forecast",
        page_icon="🌀",
        layout="wide",
    )

    st.title("🌀 ChakraNetra — AI Cyclone Forecasting")
    st.caption("SIH Hackathon Prototype | North Indian Ocean Basin")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Settings")

        # Mode toggle
        use_api = st.checkbox(
            "Use API backend",
            value=False,
            help="If checked, calls http://localhost:8000. Otherwise imports modules directly.",
        )

        api_up = False
        if use_api:
            api_up = _api_available(API_BASE)
            if api_up:
                st.success(f"API connected: {API_BASE}")
            else:
                st.warning(f"API unreachable at {API_BASE}. Falling back to direct imports.")
                use_api = False

        if not use_api:
            st.info("Mode: Direct import (no API)")

        st.divider()

        # Storm selector
        st.subheader("Select Storm")
        try:
            if use_api and api_up:
                storm_ids = _get_storm_ids_api(API_BASE)
            else:
                storm_ids = _get_storm_ids_direct()
        except Exception as e:
            st.error(f"Cannot load storms: {e}")
            storm_ids = []

        if not storm_ids:
            st.error("No storms available. Run `python -m src.data_pipeline` first.")
            return

        selected_storm = st.selectbox(
            "Storm ID",
            storm_ids,
            index=0,
            help="IBTrACS Storm Identifier",
        )

        lead_times = [24, 48, 72]

    # --- Load actual track ---
    try:
        df = pd.read_csv(STORMS_CSV)
        storm_df = df[df["storm_id"] == selected_storm].copy()
        storm_df = storm_df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        st.error(f"Cannot read storms.csv: {e}")
        return

    if storm_df.empty:
        st.warning(f"No data for storm {selected_storm}")
        return

    # --- Get prediction ---
    prediction = None
    with st.spinner("Running forecast..."):
        try:
            if use_api and api_up:
                prediction = _predict_api(API_BASE, selected_storm, lead_times)
            else:
                prediction = _predict_direct(selected_storm, lead_times)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # --- Main layout: map + sidebar table ---
    col_map, col_info = st.columns([3, 1])

    with col_map:
        m = build_map(storm_df, prediction)
        st_folium(m, width=None, height=550, returned_objects=[])

    with col_info:
        # Storm summary
        st.subheader(f"Storm: {selected_storm}")
        st.metric("Basin", storm_df.iloc[0].get("basin", "NI"))
        st.metric("Observations", len(storm_df))
        peak = storm_df["wind_kt"].max()
        st.metric("Peak Wind", f"{peak:.0f} kt")

        if prediction:
            st.divider()

            # Intensity table with calibrated intervals
            st.subheader("Forecast Intensity")
            if prediction.get("intensity"):
                rows = []
                for pt in prediction["intensity"]:
                    interval = pt.get("interval_kt")
                    if interval:
                        interval_str = f"[{interval[0]:.0f}, {interval[1]:.0f}]"
                    else:
                        interval_str = "--"
                    rows.append({
                        "Lead": f"+{pt['lead_h']}h",
                        "Wind (kt)": f"{pt['wind_kt']:.1f}",
                        "80% Interval": interval_str,
                        "Pres (hPa)": f"{pt['pressure_hpa']:.0f}",
                    })
                st.table(pd.DataFrame(rows))

            # Empirical coverage
            cov = prediction.get("empirical_coverage")
            if cov and cov > 0:
                st.caption(f"Empirical coverage: {cov:.1%}")

            # Risk summary
            risk = prediction.get("risk")
            if risk:
                st.divider()
                st.subheader("Risk Assessment")
                score = risk["risk_score"]
                # Color-code risk
                if score >= 0.7:
                    st.error(f"Risk Score: {score:.2f}")
                elif score >= 0.4:
                    st.warning(f"Risk Score: {score:.2f}")
                else:
                    st.success(f"Risk Score: {score:.2f}")

                radii = risk.get("wind_radii_km", {})
                r_rows = []
                for key in ["34kt", "50kt", "64kt"]:
                    r = radii.get(key, 0)
                    r_rows.append({"Threshold": key, "Radius (km)": f"{r:.0f}" if r > 0 else "--"})
                st.table(pd.DataFrame(r_rows))

            # Model version
            mv = prediction.get("model_version", "unknown")
            st.caption(f"Model: {mv}")

    # --- Legend ---
    st.markdown(
        """
        <div style="font-size: 0.8em; color: #666; margin-top: 8px;">
        <b>Legend:</b>
        <span style="color: #2563EB;">● Actual track</span> |
        <span style="color: #DC2626;">▲ Predicted track</span> |
        <span style="color: #FCA5A5;">○ Uncertainty cone</span> |
        <span style="color: #FDE68A;">○ 34kt</span>
        <span style="color: #FDBA74;">○ 50kt</span>
        <span style="color: #F87171;">○ 64kt</span> wind radii
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
