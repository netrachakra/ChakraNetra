"""
app.py -- ChakraNetra Streamlit Dashboard
Team Techtonic | SIH 2026 (SIH26070)

Single-screen cyclone forecasting demo with:
  - Metric cards (Storm ID, Basin, Peak Wind, Risk tier)
  - Tabs: Track & Forecast (map + charts) | Risk (wind radii + gauge)
  - Trust footer from PIPELINE_STATUS.md
  - API or direct-import fallback via sidebar

Data source: FastAPI at http://localhost:8000 or direct src/ imports.
"""

import math
import os
import sys

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

STORMS_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "storms.csv")
API_BASE = os.environ.get("CHAKRANETRA_API_URL", "http://localhost:8000")

# --------------------------------------------------------------------------- #
# Design tokens
# --------------------------------------------------------------------------- #
ACCENT = "#06B6D4"        # cyan-500 -- primary accent
ACCENT_DIM = "#164E63"    # cyan-900 -- muted accent
BG_CARD = "#1E293B"       # slate-800
BG_SURFACE = "#0F172A"    # slate-900
TEXT_PRIMARY = "#F1F5F9"   # slate-100
TEXT_MUTED = "#94A3B8"     # slate-400

# Saffir-Simpson color scale (consistent everywhere)
CAT_COLORS = {
    "TD":    "#6B7280",   # gray
    "TS":    "#22D3EE",   # cyan
    "Cat 1": "#FACC15",   # yellow
    "Cat 2": "#F97316",   # orange
    "Cat 3": "#EF4444",   # red
    "Cat 4": "#DC2626",   # dark red
    "Cat 5": "#A855F7",   # purple
}

CONE_COLORS = ["rgba(251,191,36,0.25)", "rgba(251,146,60,0.20)", "rgba(239,68,68,0.15)"]

# Map element colors
ACTUAL_TRACK = "#3B82F6"
PRED_TRACK = "#F43F5E"
CONE_FILL = "#FB923C"
WIND_34 = "#FACC15"
WIND_50 = "#F97316"
WIND_64 = "#EF4444"


def _wind_category(kt: float) -> str:
    if kt >= 137: return "Cat 5"
    if kt >= 113: return "Cat 4"
    if kt >= 96:  return "Cat 3"
    if kt >= 83:  return "Cat 2"
    if kt >= 64:  return "Cat 1"
    if kt >= 34:  return "TS"
    return "TD"


def _wind_color(kt: float) -> str:
    return CAT_COLORS.get(_wind_category(kt), "#6B7280")


def _risk_tier(score: float) -> tuple[str, str]:
    """Return (label, hex_color) for a risk score."""
    if score >= 0.85: return "Extreme", "#A855F7"
    if score >= 0.70: return "Severe",  "#EF4444"
    if score >= 0.55: return "High",    "#F97316"
    if score >= 0.40: return "Moderate","#FACC15"
    if score >= 0.20: return "Low",     "#22D3EE"
    return "Minimal", "#6B7280"


# --------------------------------------------------------------------------- #
# CSS injection
# --------------------------------------------------------------------------- #
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global */
.stApp { font-family: 'Inter', sans-serif; }
code, .stCode { font-family: 'JetBrains Mono', monospace; }

/* Metric cards */
div[data-testid="stMetric"] {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 12px 16px;
}
div[data-testid="stMetric"] label { color: #94A3B8 !important; font-size: 0.75rem !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #F1F5F9 !important; font-weight: 600 !important; }

/* Tab styling */
button[data-baseweb="tab"] { font-weight: 500; }

/* Tables */
.stTable table { border-collapse: collapse; width: 100%; }
.stTable th { background: #1E293B; color: #94A3B8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
.stTable td { color: #E2E8F0; border-bottom: 1px solid #334155; }

/* Hide default hamburger menu + footer for clean demo, but keep header for sidebar toggle */
#MainMenu, footer { visibility: hidden; }
</style>
"""


# --------------------------------------------------------------------------- #
# Backend communication
# --------------------------------------------------------------------------- #

def _api_available(base_url: str) -> bool:
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
    from src.model import predict_track_intensity
    from src.calibration import calibrate
    from src.risk import compute_risk

    raw = predict_track_intensity(storm_id, lead_times)
    cal = calibrate(raw)

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

    return {
        "storm_id": cal["storm_id"],
        "track": cal["track"],
        "intensity": cal["intensity"],
        "empirical_coverage": cal.get("empirical_coverage"),
        "risk": risk_result,
        "model_version": "direct-import",
    }


def _predict_from_upload(history_df: pd.DataFrame, lead_times: list[int]) -> dict:
    """Chain predict_from_history -> calibrate -> risk for uploaded storms."""
    from src.model import predict_from_history
    from src.calibration import calibrate
    from src.risk import compute_risk

    raw = predict_from_history(history_df, lead_times)
    cal = calibrate(raw)

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

    return {
        "storm_id": cal["storm_id"],
        "track": cal["track"],
        "intensity": cal["intensity"],
        "empirical_coverage": cal.get("empirical_coverage"),
        "risk": risk_result,
        "model_version": "direct-import (uploaded)",
    }



# --------------------------------------------------------------------------- #
# Map
# --------------------------------------------------------------------------- #

def _bearing_point(lat, lon, distance_km, bearing_deg):
    """Compute a point at a given distance and bearing from origin."""
    R = 6371.0
    d = distance_km / R
    b = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(b))
    lon2 = lon1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(lat1),
                              math.cos(d) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)


def _cone_polygon_coords(lat, lon, radius_km, n_points=36):
    """Generate a circle polygon (list of [lat,lon]) for the uncertainty cone."""
    coords = []
    for i in range(n_points + 1):
        bearing = 360.0 * i / n_points
        plat, plon = _bearing_point(lat, lon, radius_km, bearing)
        coords.append([plat, plon])
    return coords


def build_map(storm_df: pd.DataFrame, prediction: dict) -> folium.Map:
    """Build the forecast map with actual + predicted tracks and overlays."""
    center_lat = storm_df["lat"].mean()
    center_lon = storm_df["lon"].mean()

    # BUG FIX #1: Use direct CARTO CDN URL (free, no API key).
    # The named presets ("cartodbdark_matter", "CartoDB positron") route
    # through the keyed api.carto.com on some folium/system versions.
    # This URL hits basemaps.cartocdn.com directly — always free.
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles="https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr=(
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
            'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        ),
    )

    # --- Actual track: intensity-colored segments ---
    for i in range(len(storm_df) - 1):
        row = storm_df.iloc[i]
        nxt = storm_df.iloc[i + 1]
        color = _wind_color(row["wind_kt"])
        folium.PolyLine(
            [[row["lat"], row["lon"]], [nxt["lat"], nxt["lon"]]],
            color=color, weight=3, opacity=0.9,
            tooltip=f"Actual | {row['timestamp']} | {row['wind_kt']:.0f} kt ({_wind_category(row['wind_kt'])})",
        ).add_to(m)

    # Observation dots
    for _, row in storm_df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            color=_wind_color(row["wind_kt"]),
            fill=True, fill_opacity=0.9, weight=1,
            tooltip=f"{row['timestamp']} | {row['wind_kt']:.0f} kt | {row['pressure_hpa']:.0f} hPa",
        ).add_to(m)

    # Genesis marker
    genesis = storm_df.iloc[0]
    folium.Marker(
        location=[genesis["lat"], genesis["lon"]],
        icon=folium.DivIcon(html=(
            '<div style="background:#3B82F6;color:white;border-radius:50%;width:20px;height:20px;'
            'display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;'
            'border:2px solid white;">G</div>'
        )),
        tooltip=f"Genesis: {genesis['timestamp']}",
    ).add_to(m)

    # --- Predicted track + cones ---
    if prediction and prediction.get("track"):
        last_actual = storm_df.iloc[-1]
        pred_coords = [[last_actual["lat"], last_actual["lon"]]]

        for pt in prediction["track"]:
            pred_coords.append([pt["lat"], pt["lon"]])

            # Predicted intensity color
            wind_at_lead = 40.0  # fallback
            if prediction.get("intensity"):
                match_int = next(
                    (ip for ip in prediction["intensity"] if ip["lead_h"] == pt["lead_h"]),
                    None,
                )
                if match_int:
                    wind_at_lead = match_int["wind_kt"]

            # Cone as shaded polygon (not circle)
            cone_upper = pt.get("cone_km_upper", 0)
            if cone_upper and cone_upper > 0:
                poly_coords = _cone_polygon_coords(pt["lat"], pt["lon"], cone_upper)
                folium.Polygon(
                    locations=poly_coords,
                    color=CONE_FILL, weight=1, opacity=0.4,
                    fill=True, fill_color=CONE_FILL, fill_opacity=0.12,
                    tooltip=f"+{pt['lead_h']}h uncertainty cone: {cone_upper:.0f} km radius",
                ).add_to(m)

            # Predicted point marker
            folium.CircleMarker(
                location=[pt["lat"], pt["lon"]],
                radius=7,
                color=PRED_TRACK, fill=True,
                fill_color=_wind_color(wind_at_lead),
                fill_opacity=0.9, weight=2,
                tooltip=(
                    f"Forecast +{pt['lead_h']}h | "
                    f"Lat: {pt['lat']:.2f}, Lon: {pt['lon']:.2f} | "
                    f"{wind_at_lead:.0f} kt ({_wind_category(wind_at_lead)})"
                ),
            ).add_to(m)

            # Lead-time label
            folium.Marker(
                location=[pt["lat"], pt["lon"]],
                icon=folium.DivIcon(html=(
                    f'<div style="color:white;font-size:9px;font-weight:600;'
                    f'text-shadow:0 0 3px black;margin-left:10px;margin-top:-5px;">'
                    f'+{pt["lead_h"]}h</div>'
                )),
            ).add_to(m)

        # Dashed predicted line
        folium.PolyLine(
            pred_coords,
            color=PRED_TRACK, weight=2, dash_array="6 4", opacity=0.8,
            tooltip="Predicted Track",
        ).add_to(m)

    # --- Wind radii circles ---
    risk = prediction.get("risk") if prediction else None
    if risk and risk.get("wind_radii_km") and prediction.get("intensity"):
        strongest = max(prediction["intensity"], key=lambda p: p["wind_kt"])
        matching = next(
            (t for t in prediction["track"] if t["lead_h"] == strongest["lead_h"]),
            None,
        )
        if matching:
            radii = risk["wind_radii_km"]
            for label, color, key in [
                ("34 kt", WIND_34, "34kt"),
                ("50 kt", WIND_50, "50kt"),
                ("64 kt", WIND_64, "64kt"),
            ]:
                r_km = radii.get(key, 0)
                if r_km > 0:
                    folium.Circle(
                        location=[matching["lat"], matching["lon"]],
                        radius=r_km * 1000,
                        color=color, weight=2, opacity=0.7,
                        fill=True, fill_color=color, fill_opacity=0.08,
                        tooltip=f"{label} wind radius: {r_km:.0f} km",
                    ).add_to(m)

    # Fit bounds
    all_lats = list(storm_df["lat"])
    all_lons = list(storm_df["lon"])
    if prediction and prediction.get("track"):
        all_lats += [pt["lat"] for pt in prediction["track"]]
        all_lons += [pt["lon"] for pt in prediction["track"]]
    m.fit_bounds([
        [min(all_lats) - 1, min(all_lons) - 1],
        [max(all_lats) + 1, max(all_lons) + 1],
    ])

    return m


# --------------------------------------------------------------------------- #
# Plotly charts
# --------------------------------------------------------------------------- #

def _build_wind_chart(prediction: dict) -> go.Figure:
    """Wind speed vs lead time with calibrated confidence band."""
    pts = prediction.get("intensity", [])
    if not pts:
        return None

    leads = [p["lead_h"] for p in pts]
    winds = [p["wind_kt"] for p in pts]
    lows = [p["interval_kt"][0] if p.get("interval_kt") else p["wind_kt"] for p in pts]
    highs = [p["interval_kt"][1] if p.get("interval_kt") else p["wind_kt"] for p in pts]

    fig = go.Figure()
    # Confidence band
    fig.add_trace(go.Scatter(
        x=leads + leads[::-1], y=highs + lows[::-1],
        fill="toself", fillcolor="rgba(6,182,212,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="80% Interval", hoverinfo="skip",
    ))
    # Point predictions
    fig.add_trace(go.Scatter(
        x=leads, y=winds, mode="lines+markers",
        line=dict(color=ACCENT, width=2),
        marker=dict(size=8, color=[_wind_color(w) for w in winds], line=dict(color="white", width=1)),
        name="Forecast",
        hovertemplate="+%{x}h: %{y:.1f} kt<extra></extra>",
    ))
    # Category thresholds
    for kt, label, color in [(34, "TS", "#22D3EE"), (64, "Cat 1", "#FACC15"), (96, "Cat 3", "#EF4444")]:
        fig.add_hline(y=kt, line_dash="dot", line_color=color, opacity=0.4,
                      annotation_text=label, annotation_position="bottom right",
                      annotation_font_color=color, annotation_font_size=10)

    fig.update_layout(
        template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=260, margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(title="Lead Time (hours)", dtick=24, gridcolor="#334155"),
        yaxis=dict(title="Wind (kt)", gridcolor="#334155"),
        legend=dict(orientation="h", y=-0.25), font=dict(family="Inter", size=12),
        title=dict(text="Wind Speed Forecast", font=dict(size=14, color=TEXT_MUTED)),
    )
    return fig


def _build_pressure_chart(prediction: dict) -> go.Figure:
    """Pressure vs lead time."""
    pts = prediction.get("intensity", [])
    if not pts:
        return None

    leads = [p["lead_h"] for p in pts]
    pres = [p["pressure_hpa"] for p in pts]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=leads, y=pres, mode="lines+markers",
        line=dict(color="#A78BFA", width=2),
        marker=dict(size=8, color="#A78BFA", line=dict(color="white", width=1)),
        name="Pressure",
        hovertemplate="+%{x}h: %{y:.0f} hPa<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=220, margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(title="Lead Time (hours)", dtick=24, gridcolor="#334155"),
        yaxis=dict(title="Pressure (hPa)", gridcolor="#334155"),
        legend=dict(orientation="h", y=-0.3), font=dict(family="Inter", size=12),
        title=dict(text="Central Pressure Forecast", font=dict(size=14, color=TEXT_MUTED)),
    )
    return fig


def _build_risk_gauge(score: float) -> go.Figure:
    """Risk score gauge."""
    tier, color = _risk_tier(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        number=dict(suffix="%", font=dict(size=36, color="white")),
        title=dict(text=tier, font=dict(size=18, color=color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#475569", dtick=20),
            bar=dict(color=color, thickness=0.3),
            bgcolor="#1E293B",
            borderwidth=0,
            steps=[
                dict(range=[0, 20],  color="#164E63"),
                dict(range=[20, 40], color="#1E3A5F"),
                dict(range=[40, 55], color="#3B2F0A"),
                dict(range=[55, 70], color="#4A1D0A"),
                dict(range=[70, 85], color="#5C0A0A"),
                dict(range=[85, 100],color="#3B0764"),
            ],
            threshold=dict(line=dict(color="white", width=2), thickness=0.8, value=score * 100),
        ),
    ))
    fig.update_layout(
        height=220, margin=dict(l=30, r=30, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color="white"),
    )
    return fig


# --------------------------------------------------------------------------- #
# Legend HTML
# --------------------------------------------------------------------------- #
MAP_LEGEND_HTML = f"""
<div style="display:flex;flex-wrap:wrap;gap:14px;align-items:center;
            padding:8px 12px;border-radius:8px;background:#1E293B;
            border:1px solid #334155;font-size:0.78rem;color:#CBD5E1;">
    <span><span style="display:inline-block;width:14px;height:3px;background:{ACTUAL_TRACK};
          border-radius:2px;vertical-align:middle;margin-right:4px;"></span>Actual Track</span>
    <span><span style="display:inline-block;width:14px;height:3px;background:{PRED_TRACK};
          border-radius:2px;vertical-align:middle;margin-right:4px;
          border-top:2px dashed {PRED_TRACK};height:0;"></span>Predicted Track</span>
    <span><span style="display:inline-block;width:12px;height:12px;background:{CONE_FILL};
          opacity:0.4;border-radius:50%;vertical-align:middle;margin-right:4px;"></span>Uncertainty Cone</span>
    <span><span style="display:inline-block;width:10px;height:10px;border:2px solid {WIND_34};
          border-radius:50%;vertical-align:middle;margin-right:3px;"></span>34 kt</span>
    <span><span style="display:inline-block;width:10px;height:10px;border:2px solid {WIND_50};
          border-radius:50%;vertical-align:middle;margin-right:3px;"></span>50 kt</span>
    <span><span style="display:inline-block;width:10px;height:10px;border:2px solid {WIND_64};
          border-radius:50%;vertical-align:middle;margin-right:3px;"></span>64 kt</span>
    <span style="margin-left:auto;color:#64748B;">Tiles: CARTO Dark Matter (free tier)</span>
</div>
"""


# --------------------------------------------------------------------------- #
# Main app
# --------------------------------------------------------------------------- #

def main():
    st.set_page_config(
        page_title="ChakraNetra | AI Cyclone Forecasting",
        page_icon="https://em-content.zobj.net/source/twitter/408/cyclone_1f300.png",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # --- Header ---
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
            <span style="font-size:2.2rem;">&#127744;</span>
            <div>
                <h1 style="margin:0;padding:0;font-size:1.8rem;font-weight:700;
                    color:{TEXT_PRIMARY};letter-spacing:-0.02em;">ChakraNetra</h1>
                <p style="margin:0;color:{TEXT_MUTED};font-size:0.85rem;">
                    AI-Powered Cyclone Track &amp; Intensity Forecasting
                    &nbsp;|&nbsp; Team Techtonic &nbsp;|&nbsp; SIH 2026</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Sidebar ---
    with st.sidebar:
        st.markdown(f"<h3 style='color:{ACCENT};margin-bottom:4px;'>Control Panel</h3>",
                    unsafe_allow_html=True)

        use_api = st.toggle("Use API backend", value=False,
                            help="Toggle between API and direct-import mode")
        api_up = False
        if use_api:
            api_up = _api_available(API_BASE)
            if api_up:
                st.success(f"Connected: {API_BASE}", icon="\U0001f7e2")
            else:
                st.warning("API unreachable. Using direct imports.", icon="\U0001f7e1")
                use_api = False
        if not use_api:
            st.info("Direct-import mode", icon="\U0001f4e6")

        st.divider()

        # --- Data source: built-in or upload ---
        data_source = st.radio(
            "Data Source",
            ["Built-in storms", "Upload IBTrACS CSV"],
            horizontal=True,
        )

        # State variables
        selected_storm = None
        storm_df = None
        prediction = None
        lead_times = [24, 48, 72]
        is_uploaded = False
        upload_warning = ""

        if data_source == "Built-in storms":
            try:
                storm_ids = (_get_storm_ids_api(API_BASE)
                             if (use_api and api_up)
                             else _get_storm_ids_direct())
            except Exception as e:
                st.error(f"Cannot load storms: {e}")
                storm_ids = []

            if not storm_ids:
                st.error("No storm data found. Run `python -m src.data_pipeline` first.")
                return

            selected_storm = st.selectbox("Select Storm", storm_ids, index=0)

        else:
            # --- Upload UI (Contract 6) ---
            st.subheader("Upload a storm (IBTrACS CSV)")
            uploaded = st.file_uploader(
                "Any IBTrACS-format CSV -- single or multi-storm export",
                type="csv",
            )
            is_uploaded = True

            if uploaded is not None:
                # Size check (20 MB)
                if uploaded.size > 200 * 1024 * 1024:
                    st.error("File too large (>200 MB). Use a smaller export.")
                    return

                try:
                    from src.data_pipeline import validate_upload
                    raw_upload = pd.read_csv(uploaded, low_memory=False)
                except Exception:
                    st.error("Couldn't read that as a CSV -- is it the raw IBTrACS export?")
                    return

                ok, upload_storms_df, error_msg = validate_upload(raw_upload)
                if not ok:
                    st.error(error_msg)
                    return

                if error_msg:
                    upload_warning = error_msg
                    st.warning(error_msg)

                # Storm picker for multi-storm files
                storm_options = (
                    upload_storms_df.groupby("storm_id")
                    .agg(
                        name=("name", "first") if "name" in upload_storms_df.columns
                             else ("storm_id", "first"),
                        basin=("basin", "first"),
                        obs=("timestamp", "count"),
                        start=("timestamp", "min"),
                        end=("timestamp", "max"),
                    )
                    .reset_index()
                )

                def _format_storm(sid):
                    row = storm_options[storm_options.storm_id == sid].iloc[0]
                    name = row["name"] if row["name"] and str(row["name"]).strip() not in ("", "NOT_NAMED", "UNNAMED") else ""
                    label = f"{name} ({sid})" if name else sid
                    return f"{label} -- {row['basin']}, {row['obs']} obs"

                selected_storm = st.selectbox(
                    "Storm found in this file",
                    storm_options["storm_id"].tolist(),
                    format_func=_format_storm,
                )
                storm_df = upload_storms_df[
                    upload_storms_df["storm_id"] == selected_storm
                ].copy()
                storm_df = storm_df.sort_values("timestamp").reset_index(drop=True)
                # Drop 'name' column before feeding to model
                if "name" in storm_df.columns:
                    storm_df = storm_df.drop(columns=["name"])

            else:
                st.info("Upload a CSV to get started.")
                return

        st.divider()
        # Intensity color scale reference
        st.markdown("<p style='color:#94A3B8;font-size:0.75rem;margin-bottom:4px;'>INTENSITY SCALE</p>",
                    unsafe_allow_html=True)
        for cat, col in CAT_COLORS.items():
            st.markdown(f"<span style='color:{col};font-size:0.8rem;'>&#9679; {cat}</span>",
                        unsafe_allow_html=True)

    # --- Load data (built-in path) ---
    if not is_uploaded:
        try:
            df = pd.read_csv(STORMS_CSV)
            storm_df = df[df["storm_id"] == selected_storm].copy()
            storm_df = storm_df.sort_values("timestamp").reset_index(drop=True)
        except Exception as e:
            st.error(f"Cannot read storms.csv: {e}")
            return

    if storm_df is None or storm_df.empty:
        st.warning(f"No observation data for storm **{selected_storm}**. Select a different storm.")
        return

    # --- Prediction ---
    with st.spinner("Running forecast pipeline..."):
        try:
            if is_uploaded:
                # Validate history before inference
                from src.model import validate_history
                ok, msg = validate_history(storm_df)
                if not ok:
                    st.warning(f"Can't forecast this storm yet: {msg}")
                    prediction = None
                else:
                    prediction = _predict_from_upload(storm_df, lead_times)
            elif use_api and api_up:
                prediction = _predict_api(API_BASE, selected_storm, lead_times)
            else:
                prediction = _predict_direct(selected_storm, lead_times)
        except Exception as e:
            st.error(f"Prediction failed: {e}")


    # --- Top metric cards ---
    peak_wind = storm_df["wind_kt"].max()
    basin = storm_df.iloc[0].get("basin", "NI")
    risk = prediction.get("risk") if prediction else None
    risk_score = risk["risk_score"] if risk else 0.0
    tier_label, tier_color = _risk_tier(risk_score)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Storm ID", selected_storm)
    c2.metric("Basin", "Bay of Bengal" if basin == "BOB" else "Arabian Sea" if basin == "ARB" else basin)
    c3.metric("Peak Wind", f"{peak_wind:.0f} kt", delta=_wind_category(peak_wind))
    c4.metric("Risk Tier", tier_label)

    # --- Tabs ---
    tab_track, tab_risk = st.tabs(["Track & Forecast", "Risk Assessment"])

    with tab_track:
        # Map row
        m = build_map(storm_df, prediction)
        # BUG FIX #2: Set explicit height and use_container_width to prevent
        # blank/black region caused by iframe height mismatch with the map component.
        st_folium(m, height=480, use_container_width=True, returned_objects=[])

        # Legend
        st.markdown(MAP_LEGEND_HTML, unsafe_allow_html=True)

        # Charts row
        if prediction and prediction.get("intensity"):
            ch1, ch2 = st.columns(2)
            with ch1:
                wfig = _build_wind_chart(prediction)
                if wfig:
                    st.plotly_chart(wfig, use_container_width=True, config={"displayModeBar": False})
            with ch2:
                pfig = _build_pressure_chart(prediction)
                if pfig:
                    st.plotly_chart(pfig, use_container_width=True, config={"displayModeBar": False})

            # Numeric table (compact, below charts)
            with st.expander("Numeric Forecast Table", expanded=False):
                rows = []
                for pt in prediction["intensity"]:
                    interval = pt.get("interval_kt")
                    rows.append({
                        "Lead Time": f"+{pt['lead_h']}h",
                        "Wind (kt)": f"{pt['wind_kt']:.1f}",
                        "80% Interval (kt)": f"[{interval[0]:.0f}, {interval[1]:.0f}]" if interval else "--",
                        "Pressure (hPa)": f"{pt['pressure_hpa']:.0f}",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                cov = prediction.get("empirical_coverage")
                if cov and cov > 0:
                    st.caption(f"Measured empirical coverage: {cov:.1%} (nominal 80%)")
                if is_uploaded:
                    st.caption(
                        "Uncertainty interval calibrated at 80.7% coverage on our "
                        "held-out test storms. That coverage guarantee is proven for "
                        "our test set, not specifically for storms outside it -- the "
                        "interval is computed the same way here, but treat it as "
                        "well-calibrated-in-general, not storm-specifically-verified."
                    )

    with tab_risk:
        if not risk:
            st.info("No risk assessment available for this storm.")
        else:
            r1, r2 = st.columns([1, 1])

            with r1:
                # Gauge
                gauge = _build_risk_gauge(risk_score)
                st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

                # Plain-language sentence
                tier_label, tier_color = _risk_tier(risk_score)
                strongest_wind = max(prediction["intensity"], key=lambda p: p["wind_kt"])["wind_kt"] \
                    if prediction and prediction.get("intensity") else 0
                st.markdown(
                    f"<p style='text-align:center;color:{tier_color};font-weight:600;font-size:1rem;'>"
                    f"{'This storm poses a ' + tier_label.lower() + ' wind-damage risk at ' + f'{strongest_wind:.0f}' + ' kt sustained winds.' if strongest_wind > 0 else ''}"
                    f"</p>",
                    unsafe_allow_html=True,
                )

            with r2:
                st.markdown(f"<h4 style='color:{TEXT_MUTED};'>Wind Radii</h4>", unsafe_allow_html=True)
                radii = risk.get("wind_radii_km", {})
                for key, label, color in [("34kt", "Tropical Storm (34 kt)", WIND_34),
                                           ("50kt", "Strong TS (50 kt)", WIND_50),
                                           ("64kt", "Hurricane (64 kt)", WIND_64)]:
                    r_km = radii.get(key, 0)
                    st.markdown(
                        f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:8px;'>"
                        f"<span style='display:inline-block;width:14px;height:14px;"
                        f"border:2px solid {color};border-radius:50%;'></span>"
                        f"<span style='color:{TEXT_PRIMARY};'>{label}:</span>"
                        f"<span style='color:{color};font-weight:600;'>"
                        f"{'%.0f km' % r_km if r_km > 0 else 'N/A (below threshold)'}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                st.divider()
                st.markdown(
                    f"<p style='color:{TEXT_MUTED};font-size:0.78rem;'>"
                    f"Wind field modeled using a modified Rankine vortex "
                    f"(decay exponent 0.5). RMW estimated via linear regression. "
                    f"Population-density weighting is not included in this prototype.</p>",
                    unsafe_allow_html=True,
                )

    # --- Trust footer ---
    with st.expander("About this model", expanded=False):
        st.markdown(
            f"""
**Data**: Real NOAA IBTrACS v04r01 (North Indian Ocean, 20 storms 2018-2023). Not synthetic.

**Model**: HistGradientBoostingRegressor (scikit-learn) -- a simple statistical baseline,
not an ensemble or dynamical model. Track errors: ~280 km (+24h) to ~736 km (+72h).
Not competitive with operational NWP forecasting.

**Calibration**: Split-conformal prediction with measured 80.7% empirical coverage (not hardcoded).

**Risk**: Modified Rankine vortex wind field. RMW from linear regression (statistical estimate,
not observed). Risk score is wind-speed-only; no population weighting.

**Not built this sprint**: Satellite imagery, CNN, Dvorak classification, Grad-CAM,
real SMS/WhatsApp alerts, population-density risk weighting.

*Model version: {prediction.get('model_version', 'N/A') if prediction else 'N/A'}*
            """,
        )

    # Model version footer
    mv = prediction.get("model_version", "") if prediction else ""
    st.markdown(
        f"<div style='text-align:center;color:#475569;font-size:0.7rem;margin-top:16px;'>"
        f"ChakraNetra v0.1.0 | {mv} | 51 tests passing"
        f"</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
