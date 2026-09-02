"""
data_pipeline.py — ChakraNetra data pipeline

Loads and cleans the real IBTrACS North Indian Ocean best-track dataset
into data/processed/storms.csv matching the CONTRACT.md schema.

Data source: NOAA IBTrACS v04r01, North Indian Ocean basin
             (data/raw/ibtracs.NI.list.v04r01.csv)
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

# --------------------------------------------------------------------------- #
# Storm selection — 20 well-observed North Indian Ocean cyclones (2018-2023)
# Chosen for: ≥25 observations, good wind/pressure data coverage,
# and sufficient track length for +24h/+48h/+72h forecasting.
# Mix of Bay of Bengal (BB) and Arabian Sea (AS) storms.
# --------------------------------------------------------------------------- #
SELECTED_SIDS = [
    "2023156N10067",  # Biparjoy (ARB, 2023) — 97 obs
    "2019296N15066",  # Kyarr (ARB, 2019) — 74 obs
    "2019301N05081",  # Maha (ARB→BOB, 2019) — 68 obs
    "2018279N11069",  # Luban (ARB, 2018) — 64 obs
    "2018314N12093",  # Gaja (BOB, 2018) — 63 obs
    "2019116N02090",  # Fani (BOB, 2019) — 62 obs
    "2019160N11073",  # Vayu (ARB, 2019) — 59 obs
    "2021267N18094",  # Gulab/Shaheen (BOB→ARB, 2021) — 56 obs
    "2019302N11118",  # Bulbul (BOB, 2019) — 47 obs
    "2020136N10088",  # Amphan (BOB, 2020) — 44 obs
    "2018141N08059",  # Mekunu (ARB, 2018) — 41 obs
    "2021133N10071",  # Tauktae (ARB, 2021) — 40 obs
    "2023129N08091",  # Mocha (BOB, 2023) — 40 obs
    "2022126N07095",  # Asani (BOB, 2022) — 35 obs
    "2018365N08115",  # Pabuk (BOB, 2018/19) — 35 obs
    "2020335N06090",  # Nivar (BOB, 2020) — 35 obs
    "2023334N08088",  # Michaung (BOB, 2023) — 34 obs
    "2018281N14088",  # Titli (BOB, 2018) — 34 obs
    "2023292N11063",  # Tej (ARB, 2023) — 33 obs
    "2018347N07089",  # Phethai (BOB, 2018) — 32 obs
]


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
    Clean raw IBTrACS into CONTRACT.md schema.

    Steps:
    1. Filter to selected storm SIDs
    2. Extract and rename columns: SID→storm_id, ISO_TIME→timestamp,
       LAT, LON, WMO_WIND→wind_kt, WMO_PRES→pressure_hpa
    3. Map SUBBASIN to basin (BB→BOB, AS→ARB)
    4. Drop rows with missing lat/lon/wind/pressure
    5. Convert types, sort, deduplicate
    """
    # Filter to selected storms
    selected = df[df["SID"].isin(SELECTED_SIDS)].copy()
    print(f"Selected {selected['SID'].nunique()} storms, {len(selected)} raw rows")

    # Extract relevant columns
    out = pd.DataFrame()
    out["storm_id"] = selected["SID"].values
    out["timestamp"] = pd.to_datetime(selected["ISO_TIME"].values)
    out["lat"] = pd.to_numeric(selected["LAT"].values, errors="coerce")
    out["lon"] = pd.to_numeric(selected["LON"].values, errors="coerce")
    out["wind_kt"] = pd.to_numeric(selected["WMO_WIND"].values, errors="coerce")
    out["pressure_hpa"] = pd.to_numeric(selected["WMO_PRES"].values, errors="coerce")

    # Map sub-basin to CONTRACT.md basin codes
    subbasin = selected["SUBBASIN"].values
    basin_map = {"BB": "BOB", "AS": "ARB", "CS": "ARB", "MM": "BOB"}
    out["basin"] = [basin_map.get(str(sb).strip(), "NI") for sb in subbasin]

    # Drop rows with missing essential fields
    before = len(out)
    out = out.dropna(subset=["lat", "lon", "wind_kt", "pressure_hpa"])
    print(f"Dropped {before - len(out)} rows with missing wind/pressure data")

    # Ensure types
    out["lat"] = out["lat"].astype(float).round(2)
    out["lon"] = out["lon"].astype(float).round(2)
    out["wind_kt"] = out["wind_kt"].astype(float).round(1)
    out["pressure_hpa"] = out["pressure_hpa"].astype(float).round(1)

    # Convert storm IDs to human-readable form
    # IBTrACS SID format: YYYYDDDNxxyyy → use basin-year-number
    # We'll create readable IDs like "BOB-2019-FANI"
    out["storm_id"] = out["storm_id"].astype(str)

    # Sort and deduplicate
    out = out.sort_values(["storm_id", "timestamp"]).reset_index(drop=True)
    out = out.drop_duplicates(subset=["storm_id", "timestamp"], keep="first")

    # Validate: only keep storms with ≥12 observations (72h at 6-hourly)
    obs_counts = out.groupby("storm_id").size()
    valid_storms = obs_counts[obs_counts >= 12].index
    dropped_storms = obs_counts[obs_counts < 12].index.tolist()
    if dropped_storms:
        print(f"Dropped {len(dropped_storms)} storms with <12 obs: {dropped_storms}")
    out = out[out["storm_id"].isin(valid_storms)].reset_index(drop=True)

    # Ensure column order matches CONTRACT.md
    out = out[["storm_id", "timestamp", "lat", "lon", "wind_kt", "pressure_hpa", "basin"]]

    # Format timestamp as ISO8601
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    return out


def load_processed_data() -> pd.DataFrame:
    """Load the processed storms.csv file."""
    if not OUTPUT_CSV.exists():
        raise FileNotFoundError(
            f"{OUTPUT_CSV} not found. Run `python -m src.data_pipeline` first."
        )
    df = pd.read_csv(OUTPUT_CSV, parse_dates=["timestamp"])
    return df


def run_pipeline():
    """Main pipeline: load IBTrACS → clean → write storms.csv."""
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

    # Summary stats
    print(f"\nLat range:  {df['lat'].min():.1f} – {df['lat'].max():.1f}")
    print(f"Lon range:  {df['lon'].min():.1f} – {df['lon'].max():.1f}")
    print(f"Wind range: {df['wind_kt'].min():.0f} – {df['wind_kt'].max():.0f} kt")
    print(f"Pres range: {df['pressure_hpa'].min():.0f} – {df['pressure_hpa'].max():.0f} hPa")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote -> {OUTPUT_CSV}")
    return df


if __name__ == "__main__":
    run_pipeline()
