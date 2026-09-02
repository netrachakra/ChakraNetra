"""
data_pipeline.py -- ChakraNetra data pipeline

Loads and cleans IBTrACS North Indian Ocean best-track data into
data/processed/storms.csv matching the CONTRACT.md schema.

CONTRACT.md Function Contract 4:
  clean_ibtracs_dataframe(raw_df) -- reusable cleaning for any IBTrACS CSV
  validate_upload(raw_df) -- validation for user-uploaded files
  build_storms_csv() -- thin wrapper for the CLI entry point

Data source: NOAA IBTrACS v04r01, North Indian Ocean basin
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_CSV = BASE_DIR / "data" / "raw" / "ibtracs.NI.list.v04r01.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_CSV = PROCESSED_DIR / "storms.csv"

# Required raw IBTrACS columns
REQUIRED_RAW_COLS = ["SID", "ISO_TIME", "LAT", "LON", "WMO_WIND", "WMO_PRES", "BASIN"]

# Basins the model was trained on
TRAINED_BASINS = {"BOB", "ARB"}

# Sub-basin to CONTRACT.md basin code mapping
BASIN_MAP = {"BB": "BOB", "AS": "ARB", "CS": "ARB", "MM": "BOB",
             "NI": "NI", "WA": "ARB", "EA": "BOB"}

# IBTrACS BASIN column to CONTRACT.md basin code mapping
IBTRACS_BASIN_MAP = {"NI": "NI", "SI": "SI", "SP": "SP", "WP": "WP",
                     "EP": "EP", "NA": "NA", "SA": "SA"}

# Min observations per storm for inference (motion vector needs 2-step diff)
MIN_HISTORY_OBS = 3

# --------------------------------------------------------------------------- #
# Storm selection for the built-in dataset (20 NI cyclones 2018-2023)
# --------------------------------------------------------------------------- #
SELECTED_SIDS = [
    "2023156N10067",  # Biparjoy (ARB, 2023)
    "2019296N15066",  # Kyarr (ARB, 2019)
    "2019301N05081",  # Maha (ARB->BOB, 2019)
    "2018279N11069",  # Luban (ARB, 2018)
    "2018314N12093",  # Gaja (BOB, 2018)
    "2019116N02090",  # Fani (BOB, 2019)
    "2019160N11073",  # Vayu (ARB, 2019)
    "2021267N18094",  # Gulab/Shaheen (BOB->ARB, 2021)
    "2019302N11118",  # Bulbul (BOB, 2019)
    "2020136N10088",  # Amphan (BOB, 2020)
    "2018141N08059",  # Mekunu (ARB, 2018)
    "2021133N10071",  # Tauktae (ARB, 2021)
    "2023129N08091",  # Mocha (BOB, 2023)
    "2022126N07095",  # Asani (BOB, 2022)
    "2018365N08115",  # Pabuk (BOB, 2018/19)
    "2020335N06090",  # Nivar (BOB, 2020)
    "2023334N08088",  # Michaung (BOB, 2023)
    "2018281N14088",  # Titli (BOB, 2018)
    "2023292N11063",  # Tej (ARB, 2023)
    "2018347N07089",  # Phethai (BOB, 2018)
]


# =========================================================================== #
#  CONTRACT.md Function Contract 4 -- clean_ibtracs_dataframe                 #
# =========================================================================== #

def clean_ibtracs_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean ANY raw IBTrACS dataframe into CONTRACT.md schema.

    Same cleaning rules used to build storms.csv -- pulled out of the
    file-reading step so it works on an uploaded file too.

    Returns DataFrame with columns:
        storm_id, timestamp, lat, lon, wind_kt, pressure_hpa, basin

    Handles real IBTrACS gotchas:
    - Row 0 units row ("deg", "kt", "mb") -- detected and dropped
    - WMO_WIND / WMO_PRES blank strings -> NaN via pd.to_numeric
    - SID is the storm_id; NAME is display-only
    - Drops rows missing lat/lon/wind/pressure
    - Deduplicates on (storm_id, timestamp)
    """
    df = raw_df.copy()

    # --- Detect and drop units row ---
    # IBTrACS files often have row 0 as units ("deg", "kt", "mb", etc.)
    # Detect it: if LAT in first row is not parseable as float, drop it
    if len(df) > 0 and "LAT" in df.columns:
        first_lat = df.iloc[0].get("LAT", "")
        try:
            float(str(first_lat).strip())
        except (ValueError, TypeError):
            df = df.iloc[1:].reset_index(drop=True)

    # --- Extract and rename columns ---
    out = pd.DataFrame()
    out["storm_id"] = df["SID"].astype(str).str.strip()
    out["timestamp"] = pd.to_datetime(df["ISO_TIME"].astype(str).str.strip(),
                                       errors="coerce")

    # Numeric columns -- handle blanks, whitespace, non-numeric
    for raw_col, out_col in [("LAT", "lat"), ("LON", "lon"),
                              ("WMO_WIND", "wind_kt"), ("WMO_PRES", "pressure_hpa")]:
        out[out_col] = pd.to_numeric(
            df[raw_col].astype(str).str.strip(), errors="coerce"
        )

    # --- Basin mapping ---
    # Try SUBBASIN first (more specific), fall back to BASIN
    if "SUBBASIN" in df.columns:
        subbasin = df["SUBBASIN"].astype(str).str.strip()
        out["basin"] = [BASIN_MAP.get(sb, "NI") for sb in subbasin]
    elif "BASIN" in df.columns:
        basin_raw = df["BASIN"].astype(str).str.strip()
        out["basin"] = [IBTRACS_BASIN_MAP.get(b, b) for b in basin_raw]
    else:
        out["basin"] = "NI"

    # Also carry NAME if present (for display in upload UI)
    if "NAME" in df.columns:
        out["name"] = df["NAME"].astype(str).str.strip()
    else:
        out["name"] = ""

    # --- Drop rows with missing essential fields ---
    out = out.dropna(subset=["lat", "lon", "wind_kt", "pressure_hpa", "timestamp"])

    # --- Type cleanup and rounding ---
    out["lat"] = out["lat"].astype(float).round(2)
    out["lon"] = out["lon"].astype(float).round(2)
    out["wind_kt"] = out["wind_kt"].astype(float).round(1)
    out["pressure_hpa"] = out["pressure_hpa"].astype(float).round(1)

    # --- Handle longitude convention ---
    # IBTrACS mixes [-180,180] and [0,360]. Normalize to [-180,180].
    mask_360 = out["lon"] > 180
    out.loc[mask_360, "lon"] = out.loc[mask_360, "lon"] - 360

    # --- Sort and deduplicate ---
    out = out.sort_values(["storm_id", "timestamp"]).reset_index(drop=True)
    out = out.drop_duplicates(subset=["storm_id", "timestamp"], keep="first")

    # --- Validate lat/lon ranges ---
    out = out[(out["lat"] >= -90) & (out["lat"] <= 90)]
    out = out[(out["lon"] >= -180) & (out["lon"] <= 180)]

    # --- Format timestamp as ISO8601 ---
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    return out


# =========================================================================== #
#  validate_upload -- validation for user-uploaded IBTrACS files              #
# =========================================================================== #

def validate_upload(raw_df: pd.DataFrame) -> tuple[bool, pd.DataFrame | None, str]:
    """
    Validate an uploaded IBTrACS-format CSV before running inference.

    Returns (ok, storms_df, error_msg):
    - ok=True, storms_df=cleaned DataFrame, error_msg="" on success
    - ok=False, storms_df=None, error_msg=user-friendly message on failure

    Checks (hard stops first, then warnings):
    1. Non-empty, has required columns
    2. Required columns present: SID, ISO_TIME, LAT, LON, WMO_WIND, WMO_PRES
    3. After cleaning, at least one storm has >= MIN_HISTORY_OBS rows
    4. Lat in [-90,90], lon consistent
    5. No duplicate (storm_id, timestamp) -- silently dedupe
    6. Basin outside trained basins -> warn (don't block)
    7. Timestamp cadence check -> warn if unusual
    """
    warnings_list = []

    # Check 1: Non-empty
    if raw_df is None or len(raw_df) == 0:
        return False, None, "Couldn't read that as a CSV -- is it the raw IBTrACS export?"

    # Check 2: Required columns
    # IBTrACS uses BASIN at the top level; SUBBASIN is optional
    required = ["SID", "ISO_TIME", "LAT", "LON", "WMO_WIND", "WMO_PRES"]
    missing = [c for c in required if c not in raw_df.columns]
    if missing:
        return (False, None,
                f"Missing column(s): {', '.join(missing)}. "
                f"This doesn't look like an IBTrACS export. "
                f"Expected columns: {', '.join(REQUIRED_RAW_COLS)}")

    # Run cleaning
    try:
        storms_df = clean_ibtracs_dataframe(raw_df)
    except Exception as e:
        return False, None, f"Error cleaning data: {e}"

    if len(storms_df) == 0:
        return (False, None,
                "No valid rows after cleaning. Check that LAT, LON, "
                "WMO_WIND, and WMO_PRES have numeric values.")

    # Check 3: At least one storm with enough observations
    obs_counts = storms_df.groupby("storm_id").size()
    valid_storms = obs_counts[obs_counts >= MIN_HISTORY_OBS].index.tolist()
    if not valid_storms:
        return (False, None,
                f"No storm in this file has enough observations "
                f"(need >= {MIN_HISTORY_OBS}). "
                f"Storm obs counts: {obs_counts.to_dict()}")

    # Keep only storms with enough observations
    storms_df = storms_df[storms_df["storm_id"].isin(valid_storms)].reset_index(drop=True)

    # Check 6: Basin outside trained basins (warn, don't block)
    unique_basins = storms_df["basin"].unique()
    outside_basins = [b for b in unique_basins if b not in TRAINED_BASINS]
    if outside_basins:
        warnings_list.append(
            f"Storm(s) from basin(s) {', '.join(outside_basins)} -- "
            f"outside the Bay of Bengal / Arabian Sea training data. "
            f"Predictions will be extrapolated.")

    # Check 7: Timestamp cadence (warn if unusual)
    for sid, grp in storms_df.groupby("storm_id"):
        ts = pd.to_datetime(grp["timestamp"])
        if len(ts) >= 2:
            diffs_h = ts.diff().dropna().dt.total_seconds() / 3600
            median_gap = diffs_h.median()
            if median_gap < 2 or median_gap > 24:
                warnings_list.append(
                    f"Storm {sid}: median observation gap is {median_gap:.0f}h "
                    f"(model trained on ~6h cadence).")
                break  # Only warn once

    # Build the warning message
    warning_msg = ""
    if warnings_list:
        warning_msg = "WARNINGS: " + " | ".join(warnings_list)

    return True, storms_df, warning_msg


# =========================================================================== #
#  Built-in dataset functions (unchanged API)                                 #
# =========================================================================== #

def load_raw_ibtracs() -> pd.DataFrame:
    """Load raw IBTrACS CSV, skipping the units row."""
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"IBTrACS file not found at {RAW_CSV}. "
            "Download from https://www.ncei.noaa.gov/products/international-best-track-archive"
        )
    df = pd.read_csv(RAW_CSV, skiprows=[1], low_memory=False)
    return df


def clean_ibtracs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw IBTrACS into CONTRACT.md schema (built-in storms only).
    Thin wrapper: filters to SELECTED_SIDS, then calls clean_ibtracs_dataframe.
    """
    # Filter to selected storms first
    selected = df[df["SID"].isin(SELECTED_SIDS)].copy()
    print(f"Selected {selected['SID'].nunique()} storms, {len(selected)} raw rows")

    # Use the shared cleaning function
    out = clean_ibtracs_dataframe(selected)

    # Drop the 'name' column (not in CONTRACT.md schema)
    if "name" in out.columns:
        out = out.drop(columns=["name"])

    # Validate: only keep storms with >= 12 observations (72h at 6-hourly)
    obs_counts = out.groupby("storm_id").size()
    valid_storms = obs_counts[obs_counts >= 12].index
    dropped_storms = obs_counts[obs_counts < 12].index.tolist()
    if dropped_storms:
        print(f"Dropped {len(dropped_storms)} storms with <12 obs: {dropped_storms}")
    out = out[out["storm_id"].isin(valid_storms)].reset_index(drop=True)

    dropped_rows = len(selected) - len(out)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows with missing wind/pressure data")

    # Ensure column order matches CONTRACT.md
    out = out[["storm_id", "timestamp", "lat", "lon", "wind_kt", "pressure_hpa", "basin"]]

    return out


def build_storms_csv(raw_path=None, out_path=None):
    """
    CONTRACT.md Function Contract 4: thin wrapper for CLI.
    Unchanged entry point for `python -m src.data_pipeline`.
    """
    raw_path = raw_path or RAW_CSV
    out_path = out_path or OUTPUT_CSV
    raw_df = pd.read_csv(raw_path, low_memory=False)
    # Skip units row if present
    if len(raw_df) > 0 and "LAT" in raw_df.columns:
        try:
            float(str(raw_df.iloc[0]["LAT"]).strip())
        except (ValueError, TypeError):
            raw_df = raw_df.iloc[1:].reset_index(drop=True)
    # Filter to selected storms and clean
    selected = raw_df[raw_df["SID"].isin(SELECTED_SIDS)].copy()
    storms_df = clean_ibtracs_dataframe(selected)
    if "name" in storms_df.columns:
        storms_df = storms_df.drop(columns=["name"])
    # Keep only storms with >= 12 obs
    obs_counts = storms_df.groupby("storm_id").size()
    storms_df = storms_df[storms_df["storm_id"].isin(
        obs_counts[obs_counts >= 12].index)].reset_index(drop=True)
    storms_df = storms_df[["storm_id", "timestamp", "lat", "lon",
                            "wind_kt", "pressure_hpa", "basin"]]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    storms_df.to_csv(out_path, index=False)
    return storms_df


def load_processed_data() -> pd.DataFrame:
    """Load the processed storms.csv file."""
    if not OUTPUT_CSV.exists():
        raise FileNotFoundError(
            f"{OUTPUT_CSV} not found. Run `python -m src.data_pipeline` first."
        )
    df = pd.read_csv(OUTPUT_CSV, parse_dates=["timestamp"])
    return df


def run_pipeline():
    """Main pipeline: load IBTrACS -> clean -> write storms.csv."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Loading IBTrACS North Indian Ocean dataset...")
    raw = load_raw_ibtracs()
    print(f"Raw dataset: {len(raw)} rows, {raw['SID'].nunique()} storms")

    print("\nCleaning and filtering...")
    df = clean_ibtracs(raw)

    n_storms = df["storm_id"].nunique()
    n_rows = len(df)
    print(f"\nFinal dataset: {n_rows} rows for {n_storms} storms")
    print(f"Storms: {sorted(df['storm_id'].unique())}")

    print(f"\nLat range:  {df['lat'].min():.1f} - {df['lat'].max():.1f}")
    print(f"Lon range:  {df['lon'].min():.1f} - {df['lon'].max():.1f}")
    print(f"Wind range: {df['wind_kt'].min():.0f} - {df['wind_kt'].max():.0f} kt")
    print(f"Pres range: {df['pressure_hpa'].min():.0f} - {df['pressure_hpa'].max():.0f} hPa")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote -> {OUTPUT_CSV}")
    return df


if __name__ == "__main__":
    run_pipeline()
