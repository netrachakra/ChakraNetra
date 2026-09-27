"""
app.py -- ChakraNetra Cyclone Warning Centre Dashboard
Team Techtonic | SIH 2026 (SIH26070)

Redesigned as an operational Cyclone Warning Centre interface.
Single-scroll layout: Status Bar > Map (hero) > Intelligence Strip >
Forecast Detail > Model Accuracy.

No tabs — everything visible on one page for 3-minute demo flow.
"""

import math
import os
import sys
from datetime import datetime

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

# --------------------------------------------------------------------------- #
# Design tokens — Operational color system
# --------------------------------------------------------------------------- #
BG_BASE = "#0B1120"       # deepest background
BG_SURFACE = "#111827"    # card/panel background
BG_ELEVATED = "#1F2937"   # hover/active
BORDER = "#1F2937"        # subtle
BORDER_ACTIVE = "#374151" # active
TEXT_PRIMARY = "#F9FAFB"  # headings
TEXT_SECONDARY = "#9CA3AF"  # labels
TEXT_MUTED = "#6B7280"    # metadata
ACCENT = "#0EA5E9"        # primary action
CRITICAL = "#EF4444"      # danger
WARNING = "#F59E0B"       # elevated risk
SUCCESS = "#10B981"       # stable/agree
INFO = "#3B82F6"          # informational

# Saffir-Simpson intensity scale
CAT_COLORS = {
    "TD":    "#6B7280",
    "TS":    "#22D3EE",
    "Cat 1": "#FACC15",
    "Cat 2": "#F97316",
    "Cat 3": "#EF4444",
    "Cat 4": "#DC2626",
    "Cat 5": "#A855F7",
}

CONE_COLORS = ["rgba(251,191,36,0.25)", "rgba(251,146,60,0.20)", "rgba(239,68,68,0.15)"]
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
    if score >= 0.85: return "EXTREME", CRITICAL
    if score >= 0.70: return "SEVERE",  "#DC2626"
    if score >= 0.55: return "HIGH",    "#F97316"
    if score >= 0.40: return "MODERATE", WARNING
    if score >= 0.20: return "LOW",     ACCENT
    return "MINIMAL", TEXT_MUTED


# --------------------------------------------------------------------------- #
# CSS — Operational Cyclone Warning Centre theme
# --------------------------------------------------------------------------- #
CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global ── */
html, body, .stApp {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: {BG_BASE} !important;
}}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}
div[data-testid="stToolbar"] {{ display: none; }}

/* ── Metric cards — compact operational ── */
div[data-testid="stMetric"] {{
    background: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 10px 14px;
}}
div[data-testid="stMetric"] label {{
    color: {TEXT_MUTED} !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
    color: {TEXT_PRIMARY} !important;
    font-weight: 600 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.1rem !important;
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: {BG_SURFACE} !important;
    border-right: 1px solid {BORDER} !important;
}}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label {{
    color: {TEXT_SECONDARY} !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}

/* ── Expanders ── */
details {{
    background: {BG_SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 6px !important;
}}
details summary {{
    color: {TEXT_SECONDARY} !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}}

/* ── Tables ── */
.stTable table {{ border-collapse: collapse; width: 100%; }}
.stTable th {{
    background: {BG_SURFACE}; color: {TEXT_MUTED};
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em;
}}
.stTable td {{ color: {TEXT_PRIMARY}; border-bottom: 1px solid {BORDER}; font-size: 0.82rem; }}

/* ── Dividers ── */
hr {{ border-color: {BORDER} !important; opacity: 0.5; }}

/* ── Links ── */
a {{ color: {ACCENT} !important; }}
</style>
"""


# --------------------------------------------------------------------------- #
# Backend helpers (preserved from working code)
# --------------------------------------------------------------------------- #

def _api_available(base_url: str) -> bool:
    try:
        import httpx
        r = httpx.get(f"{base_url}/health", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def _get_storm_ids_direct() -> list[str]:
    df = pd.read_csv(STORMS_CSV)
    return sorted(df["storm_id"].unique().tolist())


def _predict_direct(storm_id: str, lead_times: list[int]) -> dict:
    from src.model import predict_track_intensity
    from src.calibration import calibrate
    from src.risk import compute_risk
    from src.ri import compute_ri, RI_WATCH_THRESHOLD
    from src.erc import compute_erc, ERC_WATCH_THRESHOLD

    raw = predict_track_intensity(storm_id, lead_times)
    cal = calibrate(raw)

    storm_df = pd.read_csv(STORMS_CSV)
    storm_df = storm_df[storm_df["storm_id"] == storm_id].copy()
    storm_df = storm_df.sort_values("timestamp").reset_index(drop=True)

    ri_result, erc_result, alert_overlays = _compute_ri_erc(storm_df, cal)

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
                ri_flag=alert_overlays["ri_watch"],
                erc_flag=alert_overlays["erc_watch"],
                reference_rmw_source=erc_result.get("reference_rmw_source", "inner"),
            )

    return {
        "storm_id": cal["storm_id"],
        "track": cal["track"],
        "intensity": cal["intensity"],
        "empirical_coverage": cal.get("empirical_coverage"),
        "risk": risk_result,
        "ri": ri_result,
        "erc": erc_result,
        "alert_overlays": alert_overlays,
        "model_version": "direct-import",
    }


def _predict_from_upload(history_df: pd.DataFrame, lead_times: list[int]) -> dict:
    from src.model import predict_from_history
    from src.calibration import calibrate
    from src.risk import compute_risk
    from src.ri import compute_ri, RI_WATCH_THRESHOLD
    from src.erc import compute_erc, ERC_WATCH_THRESHOLD

    raw = predict_from_history(history_df, lead_times)
    cal = calibrate(raw)
    ri_result, erc_result, alert_overlays = _compute_ri_erc(history_df, cal)

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
                ri_flag=alert_overlays["ri_watch"],
                erc_flag=alert_overlays["erc_watch"],
                reference_rmw_source=erc_result.get("reference_rmw_source", "inner"),
            )

    return {
        "storm_id": cal["storm_id"],
        "track": cal["track"],
        "intensity": cal["intensity"],
        "empirical_coverage": cal.get("empirical_coverage"),
        "risk": risk_result,
        "ri": ri_result,
        "erc": erc_result,
        "alert_overlays": alert_overlays,
        "model_version": "direct-import (uploaded)",
    }


def _compute_ri_erc(storm_df: pd.DataFrame, cal: dict) -> tuple:
    from src.ri import compute_ri, RI_WATCH_THRESHOLD
    from src.erc import compute_erc, ERC_WATCH_THRESHOLD

    ri_result = {"environmental": {"probability_24h": 0.0, "signal": "TabNet"},
                 "lightning": {"probability_24h": 0.0, "signal": "XGBoost",
                               "data_source": "WWLLN_bias_corrected"},
                 "agreement": True}
    erc_result = {"direct_probability": 0.0, "synthetic_probability": None,
                  "rmw_jump_confirmed": None, "reference_rmw_source": "inner"}
    alert_overlays = {"ri_watch": False, "erc_watch": False}

    if len(storm_df) >= 3:
        try:
            storm_df_ts = storm_df.copy()
            if "timestamp" in storm_df_ts.columns:
                storm_df_ts["timestamp"] = pd.to_datetime(storm_df_ts["timestamp"])
            latest = storm_df_ts.iloc[-1]
            lat = float(latest["lat"])
            lon = float(latest["lon"])
            basin = str(latest.get("basin", "BOB"))
            month = int(pd.to_datetime(latest.get("timestamp", "2023-10-01")).month) \
                if "timestamp" in storm_df_ts.columns else 10
            current_wind = float(latest["wind_kt"])

            ri_result = compute_ri(storm_df_ts, lat, lon, month, basin)
            erc_result = compute_erc(storm_df_ts, current_wind, basin)

            env_p = ri_result["environmental"]["probability_24h"]
            ltg_p = ri_result["lightning"]["probability_24h"]
            ri_watch = env_p >= RI_WATCH_THRESHOLD or ltg_p >= RI_WATCH_THRESHOLD
            erc_watch = erc_result["direct_probability"] >= ERC_WATCH_THRESHOLD
            alert_overlays = {"ri_watch": ri_watch, "erc_watch": erc_watch}
        except Exception:
            pass

    return ri_result, erc_result, alert_overlays


# --------------------------------------------------------------------------- #
# Map builder (preserved)
# --------------------------------------------------------------------------- #

def _bearing_point(lat, lon, distance_km, bearing_deg):
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
    coords = []
    for i in range(n_points + 1):
        bearing = 360.0 * i / n_points
        plat, plon = _bearing_point(lat, lon, radius_km, bearing)
        coords.append([plat, plon])
    return coords


def build_map(storm_df: pd.DataFrame, prediction: dict) -> folium.Map:
    center_lat = storm_df["lat"].mean()
    center_lon = storm_df["lon"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles=None,
    )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, HERE, Garmin, OpenStreetMap contributors",
        name="Esri Dark",
        max_zoom=16,
    ).add_to(m)

    # Actual track segments
    for i in range(len(storm_df) - 1):
        row = storm_df.iloc[i]
        nxt = storm_df.iloc[i + 1]
        color = _wind_color(row["wind_kt"])
        folium.PolyLine(
            [[row["lat"], row["lon"]], [nxt["lat"], nxt["lon"]]],
            color=color, weight=3, opacity=0.9,
            tooltip=f"{row['timestamp']} | {row['wind_kt']:.0f} kt ({_wind_category(row['wind_kt'])})",
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

    # Predicted track + cones
    if prediction and prediction.get("track"):
        last_actual = storm_df.iloc[-1]
        pred_coords = [[last_actual["lat"], last_actual["lon"]]]

        for pt in prediction["track"]:
            pred_coords.append([pt["lat"], pt["lon"]])
            wind_at_lead = 40.0
            if prediction.get("intensity"):
                match_int = next(
                    (ip for ip in prediction["intensity"] if ip["lead_h"] == pt["lead_h"]),
                    None,
                )
                if match_int:
                    wind_at_lead = match_int["wind_kt"]

            cone_upper = pt.get("cone_km_upper", 0)
            if cone_upper and cone_upper > 0:
                poly_coords = _cone_polygon_coords(pt["lat"], pt["lon"], cone_upper)
                folium.Polygon(
                    locations=poly_coords,
                    color=CONE_FILL, weight=1, opacity=0.4,
                    fill=True, fill_color=CONE_FILL, fill_opacity=0.12,
                    tooltip=f"+{pt['lead_h']}h uncertainty cone: {cone_upper:.0f} km radius",
                ).add_to(m)

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

            folium.Marker(
                location=[pt["lat"], pt["lon"]],
                icon=folium.DivIcon(html=(
                    f'<div style="color:white;font-size:9px;font-weight:600;'
                    f'text-shadow:0 0 3px black;margin-left:10px;margin-top:-5px;">'
                    f'+{pt["lead_h"]}h</div>'
                )),
            ).add_to(m)

        folium.PolyLine(
            pred_coords,
            color=PRED_TRACK, weight=2, dash_array="6 4", opacity=0.8,
            tooltip="Predicted Track",
        ).add_to(m)

    # Wind radii circles
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
# Plotly charts (refined)
# --------------------------------------------------------------------------- #

def _build_wind_chart(prediction: dict) -> go.Figure:
    pts = prediction.get("intensity", [])
    if not pts:
        return None

    leads = [p["lead_h"] for p in pts]
    winds = [p["wind_kt"] for p in pts]
    lows = [p["interval_kt"][0] if p.get("interval_kt") else p["wind_kt"] for p in pts]
    highs = [p["interval_kt"][1] if p.get("interval_kt") else p["wind_kt"] for p in pts]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=leads + leads[::-1], y=highs + lows[::-1],
        fill="toself", fillcolor="rgba(14,165,233,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="80% Interval", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=leads, y=winds, mode="lines+markers",
        line=dict(color=ACCENT, width=2),
        marker=dict(size=7, color=[_wind_color(w) for w in winds], line=dict(color="white", width=1)),
        name="Forecast",
        hovertemplate="+%{x}h: %{y:.1f} kt<extra></extra>",
    ))
    for kt, label, color in [(34, "TS", "#22D3EE"), (64, "Cat 1", "#FACC15"), (96, "Cat 3", CRITICAL)]:
        fig.add_hline(y=kt, line_dash="dot", line_color=color, opacity=0.3,
                      annotation_text=label, annotation_position="bottom right",
                      annotation_font_color=color, annotation_font_size=10)

    fig.update_layout(
        template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=240, margin=dict(l=40, r=16, t=28, b=36),
        xaxis=dict(title="Lead Time (h)", dtick=24, gridcolor="#1F2937"),
        yaxis=dict(title="Wind (kt)", gridcolor="#1F2937"),
        legend=dict(orientation="h", y=-0.3), font=dict(family="Inter", size=11),
        title=dict(text="WIND SPEED FORECAST", font=dict(size=11, color=TEXT_MUTED)),
    )
    return fig


def _build_pressure_chart(prediction: dict) -> go.Figure:
    pts = prediction.get("intensity", [])
    if not pts:
        return None

    leads = [p["lead_h"] for p in pts]
    pres = [p["pressure_hpa"] for p in pts]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=leads, y=pres, mode="lines+markers",
        line=dict(color="#A78BFA", width=2),
        marker=dict(size=7, color="#A78BFA", line=dict(color="white", width=1)),
        name="Pressure",
        hovertemplate="+%{x}h: %{y:.0f} hPa<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=240, margin=dict(l=40, r=16, t=28, b=36),
        xaxis=dict(title="Lead Time (h)", dtick=24, gridcolor="#1F2937"),
        yaxis=dict(title="Pressure (hPa)", gridcolor="#1F2937"),
        legend=dict(orientation="h", y=-0.3), font=dict(family="Inter", size=11),
        title=dict(text="CENTRAL PRESSURE FORECAST", font=dict(size=11, color=TEXT_MUTED)),
    )
    return fig


# --------------------------------------------------------------------------- #
# Intelligence panel HTML builders
# --------------------------------------------------------------------------- #

def _severity_card(wind_kt: float, risk_score: float, rmw_source: str) -> str:
    """Compact severity indicator card."""
    cat = _wind_category(wind_kt)
    cat_color = _wind_color(wind_kt)
    tier, tier_color = _risk_tier(risk_score)
    rmw_label = "OUTER" if rmw_source == "outer" else "INNER"
    rmw_color = "#A855F7" if rmw_source == "outer" else TEXT_MUTED

    return f"""
    <div style="background:{BG_SURFACE};border:1px solid {BORDER};border-radius:6px;
                padding:14px 16px;border-left:3px solid {cat_color};">
        <div style="color:{TEXT_MUTED};font-size:0.65rem;text-transform:uppercase;
                    letter-spacing:0.08em;margin-bottom:6px;">SEVERITY</div>
        <div style="display:flex;align-items:baseline;gap:8px;">
            <span style="font-family:'JetBrains Mono',monospace;font-size:1.6rem;
                         font-weight:700;color:{cat_color};">{wind_kt:.0f}</span>
            <span style="color:{TEXT_SECONDARY};font-size:0.85rem;">kt</span>
            <span style="background:{cat_color}22;color:{cat_color};padding:2px 8px;
                         border-radius:4px;font-size:0.72rem;font-weight:600;">{cat}</span>
        </div>
        <div style="display:flex;gap:12px;margin-top:8px;">
            <span style="color:{tier_color};font-size:0.75rem;font-weight:600;">Risk: {tier}</span>
            <span style="color:{rmw_color};font-size:0.72rem;">RMW: {rmw_label}</span>
        </div>
    </div>
    """


def _ri_card(ri_data: dict, ri_watch: bool) -> str:
    """Compact RI intelligence card."""
    env_p = ri_data.get("environmental", {}).get("probability_24h", 0.0)
    ltg_p = ri_data.get("lightning", {}).get("probability_24h", 0.0)
    agreement = ri_data.get("agreement", True)

    border_color = WARNING if ri_watch else BORDER
    status_text = "RI WATCH" if ri_watch else "MONITORING"
    status_color = WARNING if ri_watch else SUCCESS
    agree_text = "AGREE" if agreement else "DIVERGE"
    agree_color = SUCCESS if agreement else WARNING

    def _bar(value, color):
        pct = min(100, max(0, value * 100))
        return (f'<div style="background:{BG_ELEVATED};border-radius:3px;height:6px;width:100%;margin-top:3px;">'
                f'<div style="background:{color};border-radius:3px;height:6px;width:{pct}%;"></div></div>')

    return f"""
    <div style="background:{BG_SURFACE};border:1px solid {border_color};border-radius:6px;
                padding:14px 16px;border-left:3px solid {border_color};">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <span style="color:{TEXT_MUTED};font-size:0.65rem;text-transform:uppercase;
                         letter-spacing:0.08em;">RAPID INTENSIFICATION</span>
            <span style="color:{status_color};font-size:0.65rem;font-weight:600;">{status_text}</span>
        </div>
        <div style="display:flex;gap:16px;">
            <div style="flex:1;">
                <div style="display:flex;justify-content:space-between;">
                    <span style="color:{TEXT_SECONDARY};font-size:0.72rem;">ENV (TabNet)</span>
                    <span style="font-family:'JetBrains Mono',monospace;color:{TEXT_PRIMARY};
                                 font-size:0.78rem;font-weight:500;">{env_p:.0%}</span>
                </div>
                {_bar(env_p, ACCENT if env_p < 0.5 else WARNING)}
            </div>
            <div style="flex:1;">
                <div style="display:flex;justify-content:space-between;">
                    <span style="color:{TEXT_SECONDARY};font-size:0.72rem;">LTG (XGBoost)</span>
                    <span style="font-family:'JetBrains Mono',monospace;color:{TEXT_PRIMARY};
                                 font-size:0.78rem;font-weight:500;">{ltg_p:.0%}</span>
                </div>
                {_bar(ltg_p, ACCENT if ltg_p < 0.5 else WARNING)}
            </div>
        </div>
        <div style="margin-top:8px;text-align:right;">
            <span style="color:{agree_color};font-size:0.65rem;font-weight:500;">Signals: {agree_text}</span>
        </div>
    </div>
    """


def _erc_card(erc_data: dict, erc_watch: bool) -> str:
    """Compact ERC intelligence card."""
    erc_prob = erc_data.get("direct_probability", 0.0)
    phase = erc_data.get("phase", "unknown")
    rmw_src = erc_data.get("reference_rmw_source", "inner")
    details = erc_data.get("details", {})

    border_color = "#A855F7" if erc_watch else BORDER
    status_text = "ERC WATCH" if erc_watch else "NO ERC"
    status_color = "#A855F7" if erc_watch else SUCCESS

    phase_labels = {
        "insufficient_data": "Insufficient data",
        "no_qualifying_peak": "Below Cat 3 threshold",
        "no_erc_pattern": "No ERC pattern",
        "pre_erc": "Pre-ERC watch",
        "possible_onset": "Possible onset",
        "active_dip": "Active intensity dip",
    }
    phase_text = phase_labels.get(phase, phase)
    peak_w = details.get("peak_wind_kt", 0)
    dip = details.get("dip_magnitude_kt", 0)

    pct = min(100, max(0, erc_prob * 100))

    return f"""
    <div style="background:{BG_SURFACE};border:1px solid {border_color};border-radius:6px;
                padding:14px 16px;border-left:3px solid {border_color};">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <span style="color:{TEXT_MUTED};font-size:0.65rem;text-transform:uppercase;
                         letter-spacing:0.08em;">EYEWALL REPLACEMENT</span>
            <span style="color:{status_color};font-size:0.65rem;font-weight:600;">{status_text}</span>
        </div>
        <div style="display:flex;align-items:baseline;gap:6px;">
            <span style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;
                         font-weight:700;color:{TEXT_PRIMARY};">{erc_prob:.0%}</span>
            <span style="color:{TEXT_SECONDARY};font-size:0.72rem;">Direct IR</span>
        </div>
        <div style="background:{BG_ELEVATED};border-radius:3px;height:6px;width:100%;margin:6px 0;">
            <div style="background:{'#A855F7' if erc_prob >= 0.6 else ACCENT};border-radius:3px;
                        height:6px;width:{pct}%;"></div>
        </div>
        <div style="color:{TEXT_MUTED};font-size:0.68rem;">
            {phase_text}{'  |  Peak: ' + str(int(peak_w)) + ' kt  |  Dip: ' + str(int(dip)) + ' kt' if peak_w > 0 else ''}
        </div>
    </div>
    """


def _wind_radii_card(risk: dict) -> str:
    """Compact wind radii card."""
    if not risk:
        return ""
    radii = risk.get("wind_radii_km", {})
    items = [
        ("R34", WIND_34, radii.get("34kt", 0)),
        ("R50", WIND_50, radii.get("50kt", 0)),
        ("R64", WIND_64, radii.get("64kt", 0)),
    ]
    rows = ""
    for label, color, val in items:
        val_text = f"{val:.0f} km" if val > 0 else "--"
        rows += (f'<div style="display:flex;justify-content:space-between;padding:3px 0;">'
                 f'<span style="color:{color};font-size:0.75rem;font-weight:500;">{label}</span>'
                 f'<span style="font-family:\'JetBrains Mono\',monospace;color:{TEXT_PRIMARY};'
                 f'font-size:0.78rem;">{val_text}</span></div>')

    return f"""
    <div style="background:{BG_SURFACE};border:1px solid {BORDER};border-radius:6px;
                padding:14px 16px;border-left:3px solid {ACCENT};">
        <div style="color:{TEXT_MUTED};font-size:0.65rem;text-transform:uppercase;
                    letter-spacing:0.08em;margin-bottom:8px;">WIND RADII</div>
        {rows}
    </div>
    """


# --------------------------------------------------------------------------- #
# Main app
# --------------------------------------------------------------------------- #

def main():
    st.set_page_config(
        page_title="ChakraNetra | Cyclone Warning Centre",
        page_icon="https://em-content.zobj.net/source/twitter/408/cyclone_1f300.png",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ================================================================== #
    # SIDEBAR — Storm selection only, no dev controls
    # ================================================================== #
    with st.sidebar:
        st.markdown(
            f"<div style='padding:8px 0 12px 0;'>"
            f"<span style='font-size:1.4rem;font-weight:700;color:{TEXT_PRIMARY};'"
            f">ChakraNetra</span><br>"
            f"<span style='color:{TEXT_MUTED};font-size:0.72rem;'>Cyclone Warning Centre</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.divider()

        data_source = st.radio(
            "DATA SOURCE",
            ["Built-in storms", "Upload IBTrACS CSV"],
            horizontal=True,
        )

        selected_storm = None
        storm_df = None
        prediction = None
        lead_times = [24, 48, 72]
        is_uploaded = False

        if data_source == "Built-in storms":
            try:
                storm_ids = _get_storm_ids_direct()
            except Exception as e:
                st.error(f"Cannot load storms: {e}")
                storm_ids = []

            if not storm_ids:
                st.error("No storm data. Run `python -m src.data_pipeline` first.")
                return

            selected_storm = st.selectbox("SELECT STORM", storm_ids, index=0)

        else:
            st.markdown(f"<p style='color:{TEXT_SECONDARY};font-size:0.78rem;'>Upload any IBTrACS-format CSV export</p>",
                        unsafe_allow_html=True)
            uploaded = st.file_uploader("CSV file", type="csv", label_visibility="collapsed")
            is_uploaded = True

            if uploaded is not None:
                if uploaded.size > 200 * 1024 * 1024:
                    st.error("File too large (>200 MB).")
                    return

                try:
                    from src.data_pipeline import validate_upload
                    raw_upload = pd.read_csv(uploaded, low_memory=False)
                except Exception:
                    st.error("Could not read CSV -- check IBTrACS format.")
                    return

                ok, upload_storms_df, error_msg = validate_upload(raw_upload)
                if not ok:
                    st.error(error_msg)
                    return
                if error_msg:
                    st.warning(error_msg)

                storm_options = (
                    upload_storms_df.groupby("storm_id")
                    .agg(
                        name=("name", "first") if "name" in upload_storms_df.columns
                             else ("storm_id", "first"),
                        basin=("basin", "first"),
                        obs=("timestamp", "count"),
                    )
                    .reset_index()
                )

                def _format_storm(sid):
                    row = storm_options[storm_options.storm_id == sid].iloc[0]
                    name = row["name"] if str(row["name"]).strip() not in ("", "NOT_NAMED", "UNNAMED") else ""
                    label = f"{name} ({sid})" if name else sid
                    return f"{label} | {row['basin']} | {row['obs']} obs"

                selected_storm = st.selectbox(
                    "SELECT STORM", storm_options["storm_id"].tolist(),
                    format_func=_format_storm,
                )
                storm_df = upload_storms_df[
                    upload_storms_df["storm_id"] == selected_storm
                ].copy()
                storm_df = storm_df.sort_values("timestamp").reset_index(drop=True)
                if "name" in storm_df.columns:
                    storm_df = storm_df.drop(columns=["name"])
            else:
                st.info("Upload a CSV to begin.")
                return

        # Sidebar intensity scale
        st.divider()
        st.markdown(f"<p style='color:{TEXT_MUTED};font-size:0.65rem;text-transform:uppercase;"
                    f"letter-spacing:0.06em;'>INTENSITY SCALE</p>",
                    unsafe_allow_html=True)
        scale_html = ""
        for cat, col in CAT_COLORS.items():
            scale_html += f"<span style='color:{col};font-size:0.75rem;margin-right:10px;'>&#9679; {cat}</span>"
        st.markdown(scale_html, unsafe_allow_html=True)

    # ================================================================== #
    # MAIN AREA — Load data + predict
    # ================================================================== #

    if not is_uploaded:
        try:
            df = pd.read_csv(STORMS_CSV)
            storm_df = df[df["storm_id"] == selected_storm].copy()
            storm_df = storm_df.sort_values("timestamp").reset_index(drop=True)
        except Exception as e:
            st.error(f"Cannot read storms.csv: {e}")
            return

    if storm_df is None or storm_df.empty:
        st.warning(f"No data for storm **{selected_storm}**.")
        return

    with st.spinner("Running forecast..."):
        try:
            if is_uploaded:
                from src.model import validate_history
                ok, msg = validate_history(storm_df)
                if not ok:
                    st.warning(f"Cannot forecast: {msg}")
                    prediction = None
                else:
                    prediction = _predict_from_upload(storm_df, lead_times)
            else:
                prediction = _predict_direct(selected_storm, lead_times)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # ================================================================== #
    # STATUS BAR — Operational context
    # ================================================================== #
    peak_wind = storm_df["wind_kt"].max()
    basin = storm_df.iloc[0].get("basin", "NI")
    basin_name = "Bay of Bengal" if basin == "BOB" else "Arabian Sea" if basin == "ARB" else basin
    cat = _wind_category(peak_wind)
    cat_color = _wind_color(peak_wind)
    risk = prediction.get("risk") if prediction else None
    risk_score = risk["risk_score"] if risk else 0.0
    tier_label, tier_color = _risk_tier(risk_score)
    alert_overlays = prediction.get("alert_overlays", {}) if prediction else {}
    ri_watch = alert_overlays.get("ri_watch", False)
    erc_watch = alert_overlays.get("erc_watch", False)
    n_obs = len(storm_df)
    last_ts = str(storm_df.iloc[-1].get("timestamp", ""))

    # Header strip — left side (storm info)
    alert_badges = ""
    if ri_watch:
        alert_badges += f'<span style="background:{WARNING};color:#000;padding:2px 10px;border-radius:4px;font-size:0.68rem;font-weight:700;margin-left:8px;">RI WATCH</span>'
    if erc_watch:
        alert_badges += f'<span style="background:#A855F7;color:#FFF;padding:2px 10px;border-radius:4px;font-size:0.68rem;font-weight:700;margin-left:8px;">ERC WATCH</span>'

    left_html = (
        f'<span style="font-size:1.1rem;font-weight:700;color:{TEXT_PRIMARY};font-family:JetBrains Mono,monospace;">{selected_storm}</span>'
        f' <span style="background:{cat_color}22;color:{cat_color};padding:2px 10px;border-radius:4px;font-size:0.72rem;font-weight:600;">{cat} | {peak_wind:.0f} kt</span>'
        f' <span style="color:{TEXT_MUTED};font-size:0.75rem;">{basin_name}</span>'
        f' <span style="background:{tier_color}22;color:{tier_color};padding:2px 10px;border-radius:4px;font-size:0.68rem;font-weight:600;">{tier_label}</span>'
        f'{alert_badges}'
    )
    right_html = (
        f'<span style="color:{TEXT_MUTED};font-size:0.68rem;">{n_obs} obs</span>'
        f' <span style="color:{TEXT_MUTED};font-size:0.68rem;">Last: {last_ts}</span>'
        f' <span style="display:inline-block;width:6px;height:6px;background:{SUCCESS};border-radius:50%;margin-left:6px;"></span>'
        f' <span style="color:{SUCCESS};font-size:0.65rem;font-weight:500;">ONLINE</span>'
    )
    status_html = f'<div style="display:flex;align-items:center;justify-content:space-between;padding:10px 0 6px 0;border-bottom:1px solid {BORDER};margin-bottom:12px;"><div>{left_html}</div><div>{right_html}</div></div>'
    st.markdown(status_html, unsafe_allow_html=True)


    # ================================================================== #
    # MAP — Hero element, dominant
    # ================================================================== #
    m = build_map(storm_df, prediction)
    st_folium(m, height=520, use_container_width=True, returned_objects=[])

    # Map legend
    legend_items = (
        f'<span><span style="display:inline-block;width:14px;height:3px;background:{ACTUAL_TRACK};border-radius:2px;vertical-align:middle;margin-right:4px;"></span>Observed</span>'
        f' <span><span style="display:inline-block;width:14px;height:0;border-top:2px dashed {PRED_TRACK};vertical-align:middle;margin-right:4px;"></span>Forecast</span>'
        f' <span><span style="display:inline-block;width:10px;height:10px;background:{CONE_FILL};opacity:0.4;border-radius:50%;vertical-align:middle;margin-right:3px;"></span>Uncertainty</span>'
        f' <span><span style="display:inline-block;width:8px;height:8px;border:2px solid {WIND_34};border-radius:50%;vertical-align:middle;margin-right:3px;"></span>R34</span>'
        f' <span><span style="display:inline-block;width:8px;height:8px;border:2px solid {WIND_50};border-radius:50%;vertical-align:middle;margin-right:3px;"></span>R50</span>'
        f' <span><span style="display:inline-block;width:8px;height:8px;border:2px solid {WIND_64};border-radius:50%;vertical-align:middle;margin-right:3px;"></span>R64</span>'
    )
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:14px;align-items:center;padding:6px 12px;border-radius:4px;background:{BG_SURFACE};border:1px solid {BORDER};font-size:0.72rem;color:{TEXT_SECONDARY};margin-top:-8px;">{legend_items}</div>',
        unsafe_allow_html=True,
    )

    # ================================================================== #
    # INTELLIGENCE STRIP — 4 compact cards in a row
    # ================================================================== #
    st.markdown(
        f'<div style="color:{TEXT_MUTED};font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;margin:16px 0 8px 0;padding-left:2px;">STORM INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )

    ri_data = prediction.get("ri", {}) if prediction else {}
    erc_data = prediction.get("erc", {}) if prediction else {}
    rmw_src = erc_data.get("reference_rmw_source", risk.get("reference_rmw_source", "inner") if risk else "inner")

    ic1, ic2, ic3, ic4 = st.columns(4)
    with ic1:
        st.markdown(_severity_card(peak_wind, risk_score, rmw_src), unsafe_allow_html=True)
    with ic2:
        st.markdown(_ri_card(ri_data, ri_watch), unsafe_allow_html=True)
    with ic3:
        st.markdown(_erc_card(erc_data, erc_watch), unsafe_allow_html=True)
    with ic4:
        st.markdown(_wind_radii_card(risk), unsafe_allow_html=True)

    # ================================================================== #
    # FORECAST DETAIL — Charts side by side
    # ================================================================== #
    with st.expander("FORECAST DETAIL", expanded=True):
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

            # Forecast table
            coverage = prediction.get("empirical_coverage")
            if coverage:
                st.markdown(
                    f"<div style='color:{TEXT_MUTED};font-size:0.72rem;padding:4px 0;'>"
                    f"Calibration: {coverage:.1%} empirical coverage (split-conformal prediction intervals)</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(f"<p style='color:{TEXT_MUTED};'>No forecast data available.</p>",
                        unsafe_allow_html=True)

    # ================================================================== #
    # MODEL ACCURACY
    # ================================================================== #
    with st.expander("MODEL ACCURACY", expanded=False):
        from src.check_accuracy import render_accuracy_tab
        render_accuracy_tab()

    # ================================================================== #
    # SYSTEM INFORMATION
    # ================================================================== #
    with st.expander("SYSTEM INFORMATION", expanded=False):
        mv = prediction.get("model_version", "N/A") if prediction else "N/A"
        st.markdown(f"""
**Data**: IBTrACS v04r01 (North Indian Ocean, 20 storms 2018-2023)

**Model**: HistGradientBoostingRegressor | Track error: ~280 km (+24h) to ~736 km (+72h)

**Calibration**: Split-conformal prediction | Empirical coverage: 80.7%

**Risk Engine**: Modified Rankine vortex | CN-014 inner/outer eyewall switching

**RI Detection**: Dual signals -- Environmental (TabNet proxy) + Lightning Burst (XGBoost/WWLLN)

**ERC Detection**: Direct IR classifier | Synthetic PMW: gated | RMW-Jump: post-hoc

*{mv} | ChakraNetra v0.3.0 | Team Techtonic | SIH 2026*
        """)

    # Version line
    st.markdown(
        f"<div style='text-align:center;color:{TEXT_MUTED};font-size:0.62rem;margin-top:16px;padding:8px 0;'"
        f">ChakraNetra v0.3.0 | Master Plan v3 | SIH26070</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
