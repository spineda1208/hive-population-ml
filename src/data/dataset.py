"""
PyTorch Dataset classes for MSPB data.

This module provides:
- HiveDataset: Base dataset for aggregated features
- HiveSequenceDataset: Dataset for temporal sequences (LSTM/Transformer)
- MultiTaskDataset: Dataset for multi-task learning
"""

from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from .loader import (
    load_sensor_data,
    load_population_annotations,
    load_phenotypic_measurements,
    load_winter_mortality,
)
from .preprocessing import create_phenotypic_dataset
from .preprocessing import (
    create_population_dataset,
    create_phenotypic_dataset,
    create_winter_mortality_dataset,
    split_by_hive,
    SENSOR_FEATURES,
)


class HiveDataset(Dataset):
    """
    PyTorch Dataset for hive prediction using aggregated features.

    This dataset provides pre-computed statistical features suitable for
    MLP and other non-sequential models.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
        task_type: str = "regression",
    ):
        """
        Args:
            df: Preprocessed DataFrame with features and labels
            target_col: Name of target column
            feature_cols: List of feature columns (auto-detected if None)
            scaler: Pre-fitted scaler for features
            fit_scaler: Whether to fit a new scaler
            task_type: 'regression' or 'classification'
        """
        self.task_type = task_type
        self.target_col = target_col

        # Auto-detect feature columns (numerical columns excluding metadata and targets)
        if feature_cols is None:
            exclude_cols = {
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
                "n_samples",
                "window_coverage",
            }
            feature_cols = [
                c
                for c in df.columns
                if c not in exclude_cols
                and df[c].dtype in [np.float64, np.int64, float, int]
            ]

        self.feature_cols = feature_cols

        # Remove rows with missing target
        df_clean = df.dropna(subset=[target_col]).copy()

        # Extract features and handle NaN
        self.X = df_clean[feature_cols].values.astype(np.float32)
        self.X = np.nan_to_num(self.X, nan=0.0)

        # Extract targets
        if task_type == "classification":
            self.y = df_clean[target_col].values.astype(np.int64)
        else:
            self.y = df_clean[target_col].values.astype(np.float32)

        # Store metadata for reference
        self.hive_ids = (
            df_clean["hive_id"].values if "hive_id" in df_clean.columns else None
        )
        self.dates = df_clean["date"].values if "date" in df_clean.columns else None

        # Scale features
        self.scaler = scaler
        if fit_scaler:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
        elif scaler is not None:
            self.X = scaler.transform(self.X)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y[idx])
        return x, y

    @property
    def num_features(self) -> int:
        return self.X.shape[1]

    @property
    def num_classes(self) -> int:
        if self.task_type == "classification":
            return len(np.unique(self.y))
        return 1


class HiveSequenceDataset(Dataset):
    """
    PyTorch Dataset for hive prediction using temporal sequences.

    This dataset provides raw time-series data suitable for LSTM and
    Transformer models.
    """

    def __init__(
        self,
        sensor_df: pd.DataFrame,
        annotations_df: pd.DataFrame,
        target_col: str,
        window_days: int = 7,
        features: Optional[List[str]] = None,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
        task_type: str = "regression",
        max_seq_len: int = 168,  # 7 days * 24 hours (hourly sampling)
        subsample_factor: int = 12,  # Subsample to hourly from 5-min data
    ):
        """
        Args:
            sensor_df: Full sensor data DataFrame
            annotations_df: Annotations with target labels
            target_col: Name of target column
            window_days: Number of days for each sequence
            features: Sensor feature columns to use
            scaler: Pre-fitted scaler
            fit_scaler: Whether to fit new scaler
            task_type: 'regression' or 'classification'
            max_seq_len: Maximum sequence length (for padding)
        """
        self.task_type = task_type
        self.target_col = target_col
        self.window_days = window_days
        self.max_seq_len = max_seq_len
        self.subsample_factor = subsample_factor

        if features is None:
            features = SENSOR_FEATURES
        self.features = [f for f in features if f in sensor_df.columns]

        # Get unique hive IDs from sensor data
        sensor_hive_ids = set(sensor_df["hive_id"].unique())

        # Build sequences for each annotation
        self.sequences = []
        self.targets = []
        self.metadata = []

        half_window = pd.Timedelta(days=window_days // 2)

        for _, row in annotations_df.iterrows():
            hive_id = row["hive_id"]

            # Match hive ID
            matched_hive_id = None
            for sensor_id in sensor_hive_ids:
                if str(sensor_id).endswith(str(hive_id)[-4:]):
                    matched_hive_id = sensor_id
                    break

            if matched_hive_id is None:
                continue

            target_date = row[target_col] if target_col == "date" else row.get("date")
            if pd.isna(target_date):
                continue

            # Make timezone-aware if needed
            if sensor_df["timestamp"].dt.tz is not None:
                if not hasattr(target_date, "tz") or target_date.tz is None:
                    target_date = pd.Timestamp(target_date).tz_localize("UTC")

            # Get window data
            start_date = target_date - half_window
            end_date = target_date + half_window

            hive_data = sensor_df[
                (sensor_df["hive_id"] == matched_hive_id)
                & (sensor_df["timestamp"] >= start_date)
                & (sensor_df["timestamp"] <= end_date)
            ].sort_values("timestamp")

            if len(hive_data) < 10:
                continue

            # Extract sequence and subsample
            seq = hive_data[self.features].values.astype(np.float32)
            seq = np.nan_to_num(seq, nan=0.0)

            # Subsample to reduce sequence length (e.g., hourly from 5-min data)
            if self.subsample_factor > 1 and len(seq) > self.subsample_factor:
                seq = seq[:: self.subsample_factor]

            # Get target - handle classification by computing from frames_of_bees
            if target_col == "population_class":
                # Classification: need to compute from frames_of_bees
                fob = row.get("frames_of_bees")
                if pd.isna(fob):
                    continue
                target_val = 1 if fob >= 20 else 0
            else:
                target_val = (
                    row.get(target_col)
                    if target_col in row.index
                    else row.get("frames_of_bees")
                )

            if pd.isna(target_val):
                continue

            self.sequences.append(seq)
            self.targets.append(target_val)
            self.metadata.append(
                {"hive_id": hive_id, "date": target_date, "seq_len": len(seq)}
            )

        # Convert targets
        if task_type == "classification":
            self.targets = np.array(self.targets, dtype=np.int64)
        else:
            self.targets = np.array(self.targets, dtype=np.float32)

        # Fit scaler on all sequences
        self.scaler = scaler
        if fit_scaler:
            all_data = np.vstack(self.sequences)
            self.scaler = StandardScaler()
            self.scaler.fit(all_data)

        # Apply scaling
        if self.scaler is not None:
            self.sequences = [self.scaler.transform(seq) for seq in self.sequences]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        seq = self.sequences[idx]
        target = self.targets[idx]
        seq_len = len(seq)

        # Pad or truncate to max_seq_len
        if len(seq) > self.max_seq_len:
            # Take most recent data
            seq = seq[-self.max_seq_len :]
            seq_len = self.max_seq_len
        elif len(seq) < self.max_seq_len:
            # Pad with zeros
            padding = np.zeros(
                (self.max_seq_len - len(seq), seq.shape[1]), dtype=np.float32
            )
            seq = np.vstack([padding, seq])

        x = torch.from_numpy(seq.astype(np.float32))
        y = torch.tensor(target)

        return x, y, seq_len

    @property
    def num_features(self) -> int:
        return len(self.features)

    @property
    def num_classes(self) -> int:
        if self.task_type == "classification":
            return len(np.unique(self.targets))
        return 1


class MultiTaskDataset(Dataset):
    """
    Dataset for multi-task learning on multiple phenotypic traits.

    Supports predicting multiple targets simultaneously:
    - Population (regression + classification)
    - Honey yield (regression)
    - Varroa infestation (regression)
    - Defensive behavior (regression)
    - Hygienic behavior (regression)
    - Winter survival (classification)
    """

    # Task definitions
    TASK_CONFIGS = {
        "population_reg": {"col": "frames_of_bees", "type": "regression"},
        "population_cls": {"col": "population_class", "type": "classification"},
        "honey_yield": {"col": "honey_yield_kg", "type": "regression"},
        "varroa": {"col": "varroa_avg", "type": "regression"},
        "defensive": {"col": "defensive_avg", "type": "regression"},
        "hygienic": {"col": "hygienic_avg", "type": "regression"},
        "winter_survival": {"col": "survived", "type": "classification"},
    }

    def __init__(
        self,
        df: pd.DataFrame,
        tasks: List[str],
        feature_cols: Optional[List[str]] = None,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
    ):
        """
        Args:
            df: Preprocessed DataFrame with all features and labels
            tasks: List of task names to include
            feature_cols: Feature columns (auto-detected if None)
            scaler: Pre-fitted scaler
            fit_scaler: Whether to fit new scaler
        """
        self.tasks = tasks
        self.task_configs = {t: self.TASK_CONFIGS[t] for t in tasks}

        # Get target columns
        target_cols = [self.task_configs[t]["col"] for t in tasks]

        # Auto-detect feature columns
        if feature_cols is None:
            exclude_cols = set(target_cols) | {
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
                "n_samples",
                "window_coverage",
                "frames_of_bees",
                "capped_brood",
                "uncapped_brood",
                "varroa_may",
                "varroa_aug",
                "defensive_1",
                "defensive_2",
                "hygienic_1",
                "hygienic_2",
            }
            feature_cols = [
                c
                for c in df.columns
                if c not in exclude_cols
                and df[c].dtype in [np.float64, np.int64, float, int]
            ]

        self.feature_cols = feature_cols

        # Remove rows with all targets missing
        df_clean = df.dropna(subset=target_cols, how="all").copy()

        # Extract features
        self.X = df_clean[feature_cols].values.astype(np.float32)
        self.X = np.nan_to_num(self.X, nan=0.0)

        # Extract targets for each task
        self.targets = {}
        self.masks = {}  # Masks for missing values

        for task in tasks:
            col = self.task_configs[task]["col"]
            task_type = self.task_configs[task]["type"]

            values = df_clean[col].values
            mask = ~pd.isna(df_clean[col]).values

            if task_type == "classification":
                # Fill NaN with -1 for masking
                values = np.where(mask, values, -1).astype(np.int64)
            else:
                # Fill NaN with 0 for masking
                values = np.where(mask, values, 0.0).astype(np.float32)

            self.targets[task] = values
            self.masks[task] = mask.astype(np.float32)

        # Scale features
        self.scaler = scaler
        if fit_scaler:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
        elif scaler is not None:
            self.X = scaler.transform(self.X)

        # Store metadata
        self.hive_ids = (
            df_clean["hive_id"].values if "hive_id" in df_clean.columns else None
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result = {
            "features": torch.from_numpy(self.X[idx]),
        }

        for task in self.tasks:
            result[f"{task}_target"] = torch.tensor(self.targets[task][idx])
            result[f"{task}_mask"] = torch.tensor(self.masks[task][idx])

        return result

    @property
    def num_features(self) -> int:
        return self.X.shape[1]


def create_dataloaders(
    data_path: Optional[Path] = None,
    task: str = "population",
    task_type: str = "regression",
    batch_size: int = 32,
    window_days: int = 7,
    use_sequences: bool = False,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train, validation, and test DataLoaders.

    Args:
        data_path: Path to MSPB data
        task: Task name ('population', 'phenotypic', 'winter')
        task_type: 'regression' or 'classification'
        batch_size: Batch size
        window_days: Window size for temporal features
        use_sequences: Whether to use sequence dataset (for LSTM/Transformer)
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader, test_loader, info_dict)
    """
    # Load data
    sensor_df = load_sensor_data(data_path)

    # Determine target column and create appropriate dataset
    if task == "population":
        annotations_df = load_population_annotations(data_path)
        target_col = (
            "population_class" if task_type == "classification" else "frames_of_bees"
        )

        if use_sequences:
            # Create sequence datasets
            train_annot, val_annot, test_annot = split_by_hive(annotations_df)

            train_ds = HiveSequenceDataset(
                sensor_df,
                train_annot,
                target_col,
                window_days=window_days,
                fit_scaler=True,
                task_type=task_type,
            )
            val_ds = HiveSequenceDataset(
                sensor_df,
                val_annot,
                target_col,
                window_days=window_days,
                scaler=train_ds.scaler,
                task_type=task_type,
            )
            test_ds = HiveSequenceDataset(
                sensor_df,
                test_annot,
                target_col,
                window_days=window_days,
                scaler=train_ds.scaler,
                task_type=task_type,
            )
        else:
            # Create aggregated datasets
            from .preprocessing import create_population_dataset

            full_df = create_population_dataset(sensor_df, annotations_df, window_days)
            train_df, val_df, test_df = split_by_hive(full_df)

            train_ds = HiveDataset(
                train_df, target_col, fit_scaler=True, task_type=task_type
            )
            val_ds = HiveDataset(
                val_df, target_col, scaler=train_ds.scaler, task_type=task_type
            )
            test_ds = HiveDataset(
                test_df, target_col, scaler=train_ds.scaler, task_type=task_type
            )

    elif task == "winter":
        winter_df = load_winter_mortality(data_path)
        from .preprocessing import create_winter_mortality_dataset

        full_df = create_winter_mortality_dataset(sensor_df, winter_df)
        target_col = "survived"

        train_df, val_df, test_df = split_by_hive(full_df)
        train_ds = HiveDataset(
            train_df, target_col, fit_scaler=True, task_type="classification"
        )
        val_ds = HiveDataset(
            val_df, target_col, scaler=train_ds.scaler, task_type="classification"
        )
        test_ds = HiveDataset(
            test_df, target_col, scaler=train_ds.scaler, task_type="classification"
        )

    elif task in ("varroa", "honey_yield", "defensive", "hygienic"):
        from .preprocessing import create_phenotypic_dataset

        phenotypic_df = load_phenotypic_measurements(data_path)
        population_df = load_population_annotations(data_path)

        full_df = create_phenotypic_dataset(
            sensor_df, phenotypic_df, population_df, window_days
        )

        target_col_map = {
            "varroa": "varroa_avg",
            "honey_yield": "honey_yield_kg",
            "defensive": "defensive_avg",
            "hygienic": "hygienic_avg",
        }
        target_col = target_col_map[task]

        train_df, val_df, test_df = split_by_hive(full_df)
        train_ds = HiveDataset(
            train_df, target_col, fit_scaler=True, task_type="regression"
        )
        val_ds = HiveDataset(
            val_df, target_col, scaler=train_ds.scaler, task_type="regression"
        )
        test_ds = HiveDataset(
            test_df, target_col, scaler=train_ds.scaler, task_type="regression"
        )

    else:
        raise ValueError(f"Unknown task: {task}")

    # Create DataLoaders
    def collate_fn(batch):
        if use_sequences:
            # Handle variable length sequences
            x_list, y_list, len_list = zip(*batch)
            x = torch.stack(x_list)
            y = torch.stack(y_list)
            lengths = torch.tensor(len_list)
            return x, y, lengths
        else:
            x_list, y_list = zip(*batch)
            return torch.stack(x_list), torch.stack(y_list)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    info = {
        "num_features": train_ds.num_features,
        "num_classes": train_ds.num_classes,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "scaler": train_ds.scaler,
        "feature_cols": train_ds.feature_cols
        if hasattr(train_ds, "feature_cols")
        else train_ds.features,
    }

    return train_loader, val_loader, test_loader, info


if __name__ == "__main__":
    print("Testing HiveDataset (aggregated features)...")
    train_loader, val_loader, test_loader, info = create_dataloaders(
        task="population", task_type="classification", batch_size=16
    )
    print(f"Train batches: {len(train_loader)}, samples: {info['train_size']}")
    print(f"Num features: {info['num_features']}, Num classes: {info['num_classes']}")

    # Test a batch
    for x, y in train_loader:
        print(f"Batch shape: x={x.shape}, y={y.shape}")
        break

    print("\nTesting HiveSequenceDataset...")
    train_loader, val_loader, test_loader, info = create_dataloaders(
        task="population", task_type="classification", batch_size=16, use_sequences=True
    )
    print(f"Train batches: {len(train_loader)}, samples: {info['train_size']}")
    print(f"Num features: {info['num_features']}")

    for x, y, lengths in train_loader:
        print(f"Batch shape: x={x.shape}, y={y.shape}, lengths={lengths.shape}")
        break

    print("\nTesting winter mortality task...")
    train_loader, val_loader, test_loader, info = create_dataloaders(
        task="winter", batch_size=8
    )
    print(
        f"Train: {info['train_size']}, Val: {info['val_size']}, Test: {info['test_size']}"
    )
