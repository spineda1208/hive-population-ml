"""
Data preprocessing and feature engineering for MSPB dataset.

This module handles:
- Temporal aggregation of sensor data
- Feature extraction (statistics, deltas)
- Data alignment between sensor readings and annotations
- Train/val/test splitting
"""

from typing import Optional, Tuple, List, Dict
from datetime import timedelta

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# Feature columns from sensor data
AUDIO_FEATURES = [
    "hive_power",
    "audio_density",
    "audio_density_ratio",
    "density_variation",
]

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

SENSOR_FEATURES = ["temperature", "humidity"] + AUDIO_FEATURES + FREQUENCY_BANDS


def extract_temporal_features(
    sensor_df: pd.DataFrame,
    hive_id: int,
    target_date: pd.Timestamp,
    window_days: int = 7,
    features: Optional[List[str]] = None,
) -> Optional[Dict[str, float]]:
    """
    Extract statistical features from sensor data around a target date.

    Args:
        sensor_df: Full sensor DataFrame
        hive_id: Hive ID to extract features for
        target_date: Center date for the window
        window_days: Total window size in days (centered on target_date)
        features: List of feature columns to extract (defaults to SENSOR_FEATURES)

    Returns:
        Dictionary of feature statistics or None if insufficient data
    """
    if features is None:
        features = SENSOR_FEATURES

    # Filter to hive
    hive_data = sensor_df[sensor_df["hive_id"] == hive_id].copy()

    if len(hive_data) == 0:
        return None

    # Ensure timestamp is datetime
    if hive_data["timestamp"].dtype == "object":
        hive_data["timestamp"] = pd.to_datetime(hive_data["timestamp"])

    # Make target_date timezone-aware if sensor data is
    if hive_data["timestamp"].dt.tz is not None and target_date.tz is None:
        target_date = target_date.tz_localize("UTC")

    # Calculate window bounds
    half_window = timedelta(days=window_days // 2)
    start_date = target_date - half_window
    end_date = target_date + half_window

    # Filter to window
    mask = (hive_data["timestamp"] >= start_date) & (hive_data["timestamp"] <= end_date)
    window_data = hive_data[mask]

    if len(window_data) < 10:  # Minimum data points required
        return None

    result = {}

    for feature in features:
        if feature not in window_data.columns:
            continue

        values = window_data[feature].dropna()

        if len(values) == 0:
            continue

        # Basic statistics
        result[f"{feature}_mean"] = values.mean()
        result[f"{feature}_std"] = values.std()
        result[f"{feature}_min"] = values.min()
        result[f"{feature}_max"] = values.max()
        result[f"{feature}_range"] = values.max() - values.min()

        # Higher-order statistics
        if len(values) >= 4:
            result[f"{feature}_skew"] = values.skew()
            result[f"{feature}_kurtosis"] = values.kurtosis()

        # Temporal derivatives (deltas)
        if len(values) >= 2:
            delta_1 = np.diff(values.values)
            result[f"{feature}_delta1_mean"] = delta_1.mean()
            result[f"{feature}_delta1_std"] = delta_1.std()

            if len(values) >= 3:
                delta_2 = np.diff(delta_1)
                result[f"{feature}_delta2_mean"] = delta_2.mean()
                result[f"{feature}_delta2_std"] = delta_2.std()

    # Add window metadata
    result["n_samples"] = len(window_data)
    result["window_coverage"] = len(window_data) / (
        window_days * 24 * 4
    )  # Expected ~4 readings/hour

    return result


def create_population_dataset(
    sensor_df: pd.DataFrame,
    population_df: pd.DataFrame,
    window_days: int = 7,
    min_coverage: float = 0.3,
) -> pd.DataFrame:
    """
    Create a dataset for population prediction by aligning sensor features with labels.

    Args:
        sensor_df: Sensor data DataFrame
        population_df: Population annotations DataFrame
        window_days: Window size for feature extraction
        min_coverage: Minimum window coverage to include sample

    Returns:
        DataFrame with features and population labels
    """
    records = []

    # Get unique hive IDs from sensor data
    sensor_hive_ids = set(sensor_df["hive_id"].unique())

    for _, row in population_df.iterrows():
        hive_id = row["hive_id"]

        # Try to match hive ID (sensor uses full ID, annotations might use short)
        matched_hive_id = None
        for sensor_id in sensor_hive_ids:
            # Match last 4 digits
            if str(sensor_id).endswith(str(hive_id)[-4:]):
                matched_hive_id = sensor_id
                break

        if matched_hive_id is None:
            continue

        target_date = row["date"]

        features = extract_temporal_features(
            sensor_df, matched_hive_id, target_date, window_days
        )

        if features is None:
            continue

        if features.get("window_coverage", 0) < min_coverage:
            continue

        # Add labels
        features["hive_id"] = hive_id
        features["sensor_hive_id"] = matched_hive_id
        features["date"] = target_date
        features["frames_of_bees"] = row["frames_of_bees"]
        features["apiary"] = row["apiary"]
        features["evaluation"] = row.get("evaluation", "unknown")

        # Binary classification label (threshold = 20 FoB)
        features["population_class"] = 1 if row["frames_of_bees"] >= 20 else 0

        records.append(features)

    return pd.DataFrame(records)


def extract_multiscale_features(
    hive_data: pd.DataFrame,
    features: List[str],
    scales: List[str] = ["weekly", "monthly"],
) -> Dict[str, float]:
    """
    Extract features at multiple temporal scales.

    Args:
        hive_data: DataFrame with sensor data for one hive
        features: List of feature columns
        scales: List of scales to compute ('weekly', 'monthly', 'early', 'late')

    Returns:
        Dictionary of multi-scale features
    """
    result = {}

    if len(hive_data) == 0:
        return result

    hive_data = hive_data.copy()
    hive_data["week"] = hive_data["timestamp"].dt.isocalendar().week
    hive_data["month"] = hive_data["timestamp"].dt.month

    mid_idx = len(hive_data) // 2

    for feature in features:
        if feature not in hive_data.columns:
            continue
        values = hive_data[feature].dropna()
        if len(values) == 0:
            continue

        # Overall statistics (baseline)
        result[f"{feature}_overall_mean"] = values.mean()
        result[f"{feature}_overall_std"] = values.std()

        # Weekly aggregation - compute variance across weekly means
        if "weekly" in scales:
            weekly_means = hive_data.groupby("week")[feature].mean()
            if len(weekly_means) > 1:
                result[f"{feature}_weekly_trend_std"] = weekly_means.std()
                result[f"{feature}_weekly_trend_range"] = (
                    weekly_means.max() - weekly_means.min()
                )

        # Monthly aggregation
        if "monthly" in scales:
            monthly_means = hive_data.groupby("month")[feature].mean()
            if len(monthly_means) > 1:
                result[f"{feature}_monthly_trend_std"] = monthly_means.std()
                result[f"{feature}_monthly_trend_range"] = (
                    monthly_means.max() - monthly_means.min()
                )

        # Early vs late period comparison
        if "early_late" in scales or True:
            early_values = values.iloc[:mid_idx] if mid_idx > 0 else values
            late_values = values.iloc[mid_idx:] if mid_idx < len(values) else values

            if len(early_values) > 0 and len(late_values) > 0:
                result[f"{feature}_early_mean"] = early_values.mean()
                result[f"{feature}_late_mean"] = late_values.mean()
                result[f"{feature}_trend_change"] = (
                    late_values.mean() - early_values.mean()
                )

    return result


def create_phenotypic_dataset(
    sensor_df: pd.DataFrame,
    phenotypic_df: pd.DataFrame,
    population_df: pd.DataFrame,
    window_days: int = 7,
    aggregation_period: str = "summer",
    use_multiscale: bool = True,
) -> pd.DataFrame:
    """
    Create a dataset for multi-task phenotypic prediction.

    For phenotypic traits (honey yield, varroa, etc.), we aggregate sensor data
    over the summer period since these are measured once.

    Args:
        sensor_df: Sensor data DataFrame
        phenotypic_df: Phenotypic measurements DataFrame
        population_df: Population annotations (for date reference)
        window_days: Window for temporal features
        aggregation_period: 'summer' to use summer data, 'all' for all data
        use_multiscale: Whether to extract multi-scale temporal features

    Returns:
        DataFrame with features and multi-task labels
    """
    records = []

    # Get summer period dates
    if aggregation_period == "summer":
        start_date = pd.Timestamp("2020-05-01", tz="UTC")
        end_date = pd.Timestamp("2020-10-01", tz="UTC")
    else:
        start_date = sensor_df["timestamp"].min()
        end_date = sensor_df["timestamp"].max()

    sensor_hive_ids = set(sensor_df["hive_id"].unique())

    for _, row in phenotypic_df.iterrows():
        hive_id = row["hive_id"]

        # Match hive ID
        matched_hive_id = None
        for sensor_id in sensor_hive_ids:
            if str(sensor_id).endswith(str(hive_id)[-4:]):
                matched_hive_id = sensor_id
                break

        if matched_hive_id is None:
            continue

        # Get sensor data for this hive in the aggregation period
        hive_data = sensor_df[
            (sensor_df["hive_id"] == matched_hive_id)
            & (sensor_df["timestamp"] >= start_date)
            & (sensor_df["timestamp"] <= end_date)
        ]

        if len(hive_data) < 100:  # Minimum samples
            continue

        # Extract aggregate statistics
        features = {}
        for feature in SENSOR_FEATURES:
            if feature not in hive_data.columns:
                continue
            values = hive_data[feature].dropna()
            if len(values) == 0:
                continue

            features[f"{feature}_mean"] = values.mean()
            features[f"{feature}_std"] = values.std()
            features[f"{feature}_min"] = values.min()
            features[f"{feature}_max"] = values.max()

            # Daily patterns
            hive_data_copy = hive_data.copy()
            hive_data_copy["hour"] = hive_data_copy["timestamp"].dt.hour
            daily_pattern = hive_data_copy.groupby("hour")[feature].mean()
            features[f"{feature}_daily_range"] = (
                daily_pattern.max() - daily_pattern.min()
            )

        # Add multi-scale temporal features
        if use_multiscale:
            multiscale_features = extract_multiscale_features(
                hive_data, SENSOR_FEATURES, scales=["weekly", "monthly"]
            )
            features.update(multiscale_features)

        # Add labels
        features["hive_id"] = hive_id
        features["sensor_hive_id"] = matched_hive_id
        features["apiary"] = row["apiary"]

        # Target variables
        features["honey_yield_kg"] = row.get("honey_yield_kg")
        features["varroa_avg"] = row.get("varroa_avg")
        features["defensive_avg"] = row.get("defensive_avg")
        features["hygienic_avg"] = row.get("hygienic_avg")
        features["total_brood"] = row.get("total_brood")

        records.append(features)

    return pd.DataFrame(records)


def create_winter_mortality_dataset(
    sensor_df: pd.DataFrame, winter_df: pd.DataFrame, use_period: str = "pre_winter"
) -> pd.DataFrame:
    """
    Create a dataset for winter mortality prediction.

    Uses sensor data from before winter to predict survival.

    Args:
        sensor_df: Sensor data DataFrame
        winter_df: Winter mortality DataFrame
        use_period: 'pre_winter' (Sep-Nov), 'summer' (May-Aug), or 'all'

    Returns:
        DataFrame with features and survival labels
    """
    records = []

    # Define period bounds
    if use_period == "pre_winter":
        start_date = pd.Timestamp("2020-09-01", tz="UTC")
        end_date = pd.Timestamp("2020-11-14", tz="UTC")  # Before indoor wintering
    elif use_period == "summer":
        start_date = pd.Timestamp("2020-05-01", tz="UTC")
        end_date = pd.Timestamp("2020-09-01", tz="UTC")
    else:
        start_date = pd.Timestamp("2020-05-01", tz="UTC")
        end_date = pd.Timestamp("2020-11-14", tz="UTC")

    sensor_hive_ids = set(sensor_df["hive_id"].unique())

    for _, row in winter_df.iterrows():
        hive_id = row["hive_id"]

        # Match hive ID
        matched_hive_id = None
        for sensor_id in sensor_hive_ids:
            if str(sensor_id).endswith(str(hive_id)[-4:]):
                matched_hive_id = sensor_id
                break

        if matched_hive_id is None:
            continue

        # Get sensor data for period
        hive_data = sensor_df[
            (sensor_df["hive_id"] == matched_hive_id)
            & (sensor_df["timestamp"] >= start_date)
            & (sensor_df["timestamp"] <= end_date)
        ]

        if len(hive_data) < 100:
            continue

        # Extract features
        features = {}
        for feature in SENSOR_FEATURES:
            if feature not in hive_data.columns:
                continue
            values = hive_data[feature].dropna()
            if len(values) == 0:
                continue

            features[f"{feature}_mean"] = values.mean()
            features[f"{feature}_std"] = values.std()
            features[f"{feature}_min"] = values.min()
            features[f"{feature}_max"] = values.max()

        # Add metadata and labels
        features["hive_id"] = hive_id
        features["sensor_hive_id"] = matched_hive_id
        features["apiary"] = row["apiary"]
        features["survived"] = int(row["survived"])
        features["weight_before_kg"] = row.get("weight_before_kg")
        features["frames_before"] = row.get("frames_before")

        records.append(features)

    return pd.DataFrame(records)


def prepare_features_and_labels(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    scale_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler], List[str]]:
    """
    Prepare feature matrix and label vector for training.

    Args:
        df: Dataset DataFrame
        target_col: Name of target column
        feature_cols: List of feature columns (auto-detected if None)
        scale_features: Whether to standardize features

    Returns:
        Tuple of (X, y, scaler, feature_names)
    """
    # Auto-detect feature columns
    if feature_cols is None:
        exclude_cols = [
            target_col,
            "hive_id",
            "sensor_hive_id",
            "date",
            "apiary",
            "evaluation",
            "population_class",
            "survived",
            "mortality_cause",
            "honey_yield_kg",
            "varroa_avg",
            "defensive_avg",
            "hygienic_avg",
            "total_brood",
            "weight_before_kg",
            "frames_before",
            "weight_after_kg",
            "frames_after",
            "syrup_consumption_kg",
        ]
        feature_cols = [
            c
            for c in df.columns
            if c not in exclude_cols
            and df[c].dtype in [np.float64, np.int64, float, int]
        ]

    # Remove rows with missing target
    df_clean = df.dropna(subset=[target_col])

    # Get features and fill missing values
    X = df_clean[feature_cols].values.astype(np.float32)
    y = df_clean[target_col].values

    # Handle NaN in features
    X = np.nan_to_num(X, nan=0.0)

    # Scale features
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, y, scaler, feature_cols


def split_by_hive(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset by hive ID to prevent data leakage.

    Args:
        df: Dataset DataFrame with 'hive_id' column
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_state: Random seed

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    hive_ids = df["hive_id"].unique()

    # First split: train+val vs test
    train_val_hives, test_hives = train_test_split(
        hive_ids, test_size=test_size, random_state=random_state
    )

    # Second split: train vs val
    adjusted_val_size = val_size / (1 - test_size)
    train_hives, val_hives = train_test_split(
        train_val_hives, test_size=adjusted_val_size, random_state=random_state
    )

    train_df = df[df["hive_id"].isin(train_hives)]
    val_df = df[df["hive_id"].isin(val_hives)]
    test_df = df[df["hive_id"].isin(test_hives)]

    return train_df, val_df, test_df


if __name__ == "__main__":
    from loader import (
        load_sensor_data,
        load_population_annotations,
        load_phenotypic_measurements,
        load_winter_mortality,
    )

    print("Loading data...")
    sensor_df = load_sensor_data()
    pop_df = load_population_annotations()
    pheno_df = load_phenotypic_measurements()
    winter_df = load_winter_mortality()

    print("\nCreating population dataset...")
    pop_dataset = create_population_dataset(sensor_df, pop_df)
    print(f"Population dataset shape: {pop_dataset.shape}")
    print(f"Columns: {len(pop_dataset.columns)}")
    print(
        f"Class distribution: {pop_dataset['population_class'].value_counts().to_dict()}"
    )

    print("\nCreating phenotypic dataset...")
    pheno_dataset = create_phenotypic_dataset(sensor_df, pheno_df, pop_df)
    print(f"Phenotypic dataset shape: {pheno_dataset.shape}")

    print("\nCreating winter mortality dataset...")
    winter_dataset = create_winter_mortality_dataset(sensor_df, winter_df)
    print(f"Winter dataset shape: {winter_dataset.shape}")
    print(f"Survival rate: {winter_dataset['survived'].mean():.2%}")

    print("\nSplitting by hive...")
    train_df, val_df, test_df = split_by_hive(pop_dataset)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
