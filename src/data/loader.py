"""
Data loading utilities for the MSPB dataset.

This module handles loading and parsing of:
- Sensor data (CSV files with audio, temperature, humidity)
- Annotation data (Excel files with phenotypic measurements)
"""

import os
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np


# Default data path (can be overridden via environment variable)
DEFAULT_DATA_PATH = Path(
    os.environ.get("MSPB_DATA_PATH", "/Users/santiagopineda/Desktop")
)

# Sensor data columns
SENSOR_COLUMNS = {
    "timestamp": "published_at",
    "temperature": "temperature",
    "humidity": "humidity",
    "hive_id": "tag_number",
    "hive_power": "hive_power",
    "audio_density": "audio_density",
    "audio_density_ratio": "audio_density_ratio",
    "density_variation": "density_variation",
}

# Audio frequency band columns (16 bands)
FREQUENCY_BANDS = [
    "hz_122.0703125",
    "hz_152.587890625",
    "hz_183.10546875",
    "hz_213.623046875",
    "hz_244.140625",
    "hz_274.658203125",
    "hz_305.17578125",
    "hz_335.693359375",
    "hz_366.2109375",
    "hz_396.728515625",
    "hz_427.24609375",
    "hz_457.763671875",
    "hz_488.28125",
    "hz_518.798828125",
    "hz_549.31640625",
    "hz_579.833984375",
]


def load_sensor_data(
    data_path: Optional[Path] = None, period: str = "all"
) -> pd.DataFrame:
    """
    Load sensor data from CSV files.

    Args:
        data_path: Path to data directory. Defaults to MSPB_DATA_PATH env var.
        period: Which period to load - 'd1' (summer), 'd2' (winter), or 'all'.

    Returns:
        DataFrame with sensor readings indexed by timestamp and hive_id.
    """
    data_path = Path(data_path) if data_path else DEFAULT_DATA_PATH

    dfs = []

    if period in ("d1", "all"):
        d1_path = data_path / "D1_sensor_data.csv"
        if d1_path.exists():
            df_d1 = pd.read_csv(d1_path, parse_dates=["published_at"])
            df_d1["period"] = "D1"
            dfs.append(df_d1)
        else:
            raise FileNotFoundError(f"D1 sensor data not found at {d1_path}")

    if period in ("d2", "all"):
        d2_path = data_path / "D2_sensor_data.csv"
        if d2_path.exists():
            df_d2 = pd.read_csv(d2_path, parse_dates=["published_at"])
            df_d2["period"] = "D2"
            dfs.append(df_d2)
        else:
            raise FileNotFoundError(f"D2 sensor data not found at {d2_path}")

    df = pd.concat(dfs, ignore_index=True)

    # Standardize column names
    df = df.rename(columns={"published_at": "timestamp", "tag_number": "hive_id"})

    # Sort by timestamp
    df = df.sort_values(["hive_id", "timestamp"]).reset_index(drop=True)

    return df


def load_population_annotations(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load population (frames of bees) annotations from evaluation sheets.

    The evaluation sheets have a complex structure:
    - Row 0: Main headers (Dates, Yard, Hive ID, Number of boxes, Number of frames covered by bees/box)
    - Row 0 columns 4-9: Sub-headers (Brood Chamber 1, Brood chamber 2, Honey supper 1-4)
    - Row 1+: Data rows

    Total frames = sum of all individual box frame counts (columns 4-9 in data rows)

    Args:
        data_path: Path to data directory.

    Returns:
        DataFrame with columns: hive_id, date, frames_of_bees, apiary
    """
    data_path = Path(data_path) if data_path else DEFAULT_DATA_PATH

    excel_path = data_path / "D1_ant.xlsx"
    if not excel_path.exists():
        raise FileNotFoundError(f"D1 annotation file not found at {excel_path}")

    xl = pd.ExcelFile(excel_path)

    records = []

    # Process each evaluation sheet
    for sheet_name in xl.sheet_names:
        if not sheet_name.startswith("Evaluation"):
            continue

        # Read without headers to handle complex structure
        df = pd.read_excel(xl, sheet_name=sheet_name, header=None)

        # Row 0 contains sub-headers for frame counts (Brood Chamber 1, etc.)
        # Row 1+ contains actual data
        # Column structure:
        # 0: Dates, 1: Yard, 2: Hive ID, 3: Number of boxes
        # 4-9: Frame counts per box (Brood Chamber 1, Brood chamber 2, Honey supper 1-4)

        # Skip row 0 (sub-headers) and process data rows
        for row_idx in range(1, len(df)):
            row = df.iloc[row_idx]

            # Extract basic info
            date_val = row.iloc[0]
            yard = row.iloc[1]
            hive_id = row.iloc[2]

            # Skip if missing essential data
            if pd.isna(date_val) or pd.isna(hive_id):
                continue

            # Sum frames from all boxes (columns 4-9)
            # These are: Brood Chamber 1, Brood chamber 2, Honey supper 1, 2, 3, 4
            total_frames = 0
            for col_idx in range(4, min(10, len(row))):
                val = row.iloc[col_idx]
                if pd.notna(val):
                    try:
                        total_frames += float(val)
                    except (ValueError, TypeError):
                        pass

            if total_frames > 0:
                records.append(
                    {
                        "hive_id": int(float(hive_id)),
                        "date": pd.to_datetime(date_val),
                        "frames_of_bees": total_frames,
                        "apiary": str(yard).strip() if pd.notna(yard) else "Unknown",
                        "evaluation": sheet_name,
                    }
                )

    return pd.DataFrame(records)


def load_phenotypic_measurements(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load phenotypic measurements (brood, varroa, behavior, honey yield).

    Args:
        data_path: Path to data directory.

    Returns:
        DataFrame with phenotypic trait measurements per hive.
    """
    data_path = Path(data_path) if data_path else DEFAULT_DATA_PATH

    excel_path = data_path / "D1_ant.xlsx"
    if not excel_path.exists():
        raise FileNotFoundError(f"D1 annotation file not found at {excel_path}")

    df = pd.read_excel(excel_path, sheet_name="Phenotypic measurements", header=None)

    # The first two rows contain headers
    # Row 0: Category headers
    # Row 1: Specific column names

    # Find Hive ID column
    hive_id_col_idx = None
    for idx, val in enumerate(df.iloc[0]):
        if "Hive ID" in str(val):
            hive_id_col_idx = idx
            break

    # Parse the data manually due to complex header structure
    records = []

    for row_idx in range(2, len(df)):
        row = df.iloc[row_idx]

        # Get hive ID (column 3 based on structure)
        hive_id = row.iloc[3] if pd.notna(row.iloc[3]) else None
        if hive_id is None:
            continue

        # Convert hive ID format (02056 -> 2056)
        try:
            hive_id = int(str(hive_id).lstrip("0"))
        except (ValueError, TypeError):
            continue

        record = {
            "hive_id": hive_id,
            "apiary": row.iloc[1],
            # Brood surface (columns 4-6)
            "capped_brood": _safe_float(row.iloc[4]),
            "uncapped_brood": _safe_float(row.iloc[5]),
            "total_brood": _safe_float(row.iloc[6]),
            # Varroa (columns 8, 10 - two measurements)
            "varroa_may": _safe_float(row.iloc[8]),
            "varroa_aug": _safe_float(row.iloc[10]),
            # Defensive behavior (columns 12, 14 - two measurements)
            "defensive_1": _safe_float(row.iloc[12]),
            "defensive_2": _safe_float(row.iloc[14]),
            # Hygienic behavior (columns 16, 18 - two measurements)
            "hygienic_1": _safe_float(row.iloc[16]),
            "hygienic_2": _safe_float(row.iloc[18]),
            # Honey production (column 20)
            "honey_yield_kg": _safe_float(row.iloc[20]),
        }

        records.append(record)

    result_df = pd.DataFrame(records)

    # Compute averages for traits with multiple measurements
    result_df["varroa_avg"] = result_df[["varroa_may", "varroa_aug"]].mean(axis=1)
    result_df["defensive_avg"] = result_df[["defensive_1", "defensive_2"]].mean(axis=1)
    result_df["hygienic_avg"] = result_df[["hygienic_1", "hygienic_2"]].mean(axis=1)

    return result_df


def load_winter_mortality(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load winter mortality and survival data.

    Args:
        data_path: Path to data directory.

    Returns:
        DataFrame with winter survival status per hive.
    """
    data_path = Path(data_path) if data_path else DEFAULT_DATA_PATH

    excel_path = data_path / "D2_ant.xlsx"
    if not excel_path.exists():
        raise FileNotFoundError(f"D2 annotation file not found at {excel_path}")

    df = pd.read_excel(excel_path, sheet_name="Sheet1")

    # Determine survival status
    # If mortality_cause is NaN and status columns indicate "good", then survived
    df["survived"] = df["Mortality cause"].isna()

    result = pd.DataFrame(
        {
            "hive_id": df["Hive ID"],
            "apiary": df["Apiary"],
            "survived": df["survived"],
            "mortality_cause": df["Mortality cause"],
            "weight_before_kg": df["weight (kg) Nov 4 2020"],
            "weight_after_kg": df["weight (kg) Apr 5 2021"],
            "syrup_consumption_kg": df["winter syrup consuption (kg)"],
            "frames_before": df["Bees frames Oct 20"],
            "frames_after": df["Bees frames Apr 2021"],
        }
    )

    return result


def get_hive_id_mapping(data_path: Optional[Path] = None) -> dict:
    """
    Get mapping between sensor hive IDs and annotation hive IDs.

    The sensor data uses IDs like 200602, 202056, etc.
    The annotation data uses IDs like 2056, 2131, etc.

    Returns:
        Dict mapping sensor_id -> annotation_id
    """
    data_path = Path(data_path) if data_path else DEFAULT_DATA_PATH

    excel_path = data_path / "D1_ant.xlsx"
    df = pd.read_excel(excel_path, sheet_name="ID lookup table")

    # Create mapping from the lookup table
    # Colony number Nectar is the sensor ID format
    mapping = {}
    for _, row in df.iterrows():
        nectar_id = row["Colony number Nectar"]
        crsad_id = row["Colony number CRSAD"]
        if pd.notna(nectar_id) and pd.notna(crsad_id):
            # Sensor IDs are like 2056, 2131 in tag_number
            # These match directly with annotation Hive IDs
            mapping[int(nectar_id)] = int(nectar_id)

    return mapping


def _safe_float(value) -> Optional[float]:
    """Safely convert a value to float, returning None if not possible."""
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def load_all_data(data_path: Optional[Path] = None) -> dict:
    """
    Load all MSPB data into a dictionary.

    Args:
        data_path: Path to data directory.

    Returns:
        Dictionary with keys:
        - 'sensor': Sensor data DataFrame
        - 'population': Population annotations DataFrame
        - 'phenotypic': Phenotypic measurements DataFrame
        - 'winter': Winter mortality DataFrame
    """
    data_path = Path(data_path) if data_path else DEFAULT_DATA_PATH

    return {
        "sensor": load_sensor_data(data_path),
        "population": load_population_annotations(data_path),
        "phenotypic": load_phenotypic_measurements(data_path),
        "winter": load_winter_mortality(data_path),
    }


if __name__ == "__main__":
    # Test loading
    print("Loading sensor data...")
    sensor_df = load_sensor_data()
    print(f"Sensor data shape: {sensor_df.shape}")
    print(f"Hive IDs: {sensor_df['hive_id'].nunique()}")
    print(
        f"Date range: {sensor_df['timestamp'].min()} to {sensor_df['timestamp'].max()}"
    )

    print("\nLoading population annotations...")
    pop_df = load_population_annotations()
    print(f"Population data shape: {pop_df.shape}")
    print(pop_df.head())

    print("\nLoading phenotypic measurements...")
    pheno_df = load_phenotypic_measurements()
    print(f"Phenotypic data shape: {pheno_df.shape}")
    print(pheno_df.head())

    print("\nLoading winter mortality...")
    winter_df = load_winter_mortality()
    print(f"Winter data shape: {winter_df.shape}")
    print(f"Survival rate: {winter_df['survived'].mean():.2%}")
