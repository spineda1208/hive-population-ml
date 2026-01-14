#!/usr/bin/env python3
"""
Ensemble training script for Varroa and other phenotypic trait prediction.

Combines MLP with sklearn models (Ridge, RandomForest, GradientBoosting)
for better performance on small datasets.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import (
    load_sensor_data,
    load_phenotypic_measurements,
    load_population_annotations,
)
from src.data.preprocessing import create_phenotypic_dataset, split_by_hive


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ensemble for phenotypic prediction"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="varroa",
        choices=["varroa", "honey_yield", "defensive", "hygienic"],
        help="Prediction task",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Output directory"
    )
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def evaluate_regression(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {}

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "n_samples": len(y_true),
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("=" * 60)
    print("Ensemble Training for Phenotypic Prediction")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"CV Folds: {args.n_folds}")

    # Load data
    print("\nLoading data...")
    sensor_df = load_sensor_data()
    phenotypic_df = load_phenotypic_measurements()
    population_df = load_population_annotations()

    # Create dataset with multi-scale features
    full_df = create_phenotypic_dataset(
        sensor_df, phenotypic_df, population_df, use_multiscale=True
    )

    target_col_map = {
        "varroa": "varroa_avg",
        "honey_yield": "honey_yield_kg",
        "defensive": "defensive_avg",
        "hygienic": "hygienic_avg",
    }
    target_col = target_col_map[args.task]

    # Get features and target
    exclude_cols = {
        target_col,
        "hive_id",
        "sensor_hive_id",
        "apiary",
        "honey_yield_kg",
        "varroa_avg",
        "defensive_avg",
        "hygienic_avg",
        "total_brood",
    }
    feature_cols = [
        c
        for c in full_df.columns
        if c not in exclude_cols
        and full_df[c].dtype in [np.float64, np.int64, float, int]
    ]

    # Clean data
    df_clean = full_df.dropna(subset=[target_col]).copy()
    X = df_clean[feature_cols].values.astype(np.float32)
    y = df_clean[target_col].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)

    print(f"Dataset size: {len(df_clean)} samples")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"Target mean: {y.mean():.2f}, std: {y.std():.2f}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define models
    models = {
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_depth=5, min_samples_leaf=3, random_state=args.seed
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=50, max_depth=3, min_samples_leaf=3, random_state=args.seed
        ),
    }

    # Cross-validation
    print("\n" + "=" * 60)
    print("Cross-Validation Results")
    print("=" * 60)

    cv = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    results = {}
    predictions = {}

    for name, model in models.items():
        print(f"\n{name}...")

        # Cross-validated predictions
        y_pred = cross_val_predict(model, X_scaled, y, cv=cv)
        predictions[name] = y_pred

        # Evaluate
        metrics = evaluate_regression(y, y_pred)
        results[name] = metrics

        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")

    # Simple ensemble (average of all models)
    print("\n" + "-" * 40)
    print("Ensemble (Average)...")

    ensemble_pred = np.mean([predictions[name] for name in models.keys()], axis=0)
    ensemble_metrics = evaluate_regression(y, ensemble_pred)
    results["Ensemble_Average"] = ensemble_metrics

    print(f"  MAE: {ensemble_metrics['mae']:.4f}")
    print(f"  RMSE: {ensemble_metrics['rmse']:.4f}")
    print(f"  R²: {ensemble_metrics['r2']:.4f}")

    # Weighted ensemble (best models get more weight)
    print("\nEnsemble (Weighted by R²)...")

    r2_scores = [max(0.01, results[name]["r2"] + 1) for name in models.keys()]
    weights = np.array(r2_scores) / sum(r2_scores)

    weighted_pred = np.zeros_like(y)
    for i, name in enumerate(models.keys()):
        weighted_pred += weights[i] * predictions[name]

    weighted_metrics = evaluate_regression(y, weighted_pred)
    results["Ensemble_Weighted"] = weighted_metrics

    print(f"  Weights: {dict(zip(models.keys(), weights.round(3)))}")
    print(f"  MAE: {weighted_metrics['mae']:.4f}")
    print(f"  RMSE: {weighted_metrics['rmse']:.4f}")
    print(f"  R²: {weighted_metrics['r2']:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("-" * 46)

    for name, metrics in results.items():
        print(
            f"{name:<20} {metrics['mae']:>8.4f} {metrics['rmse']:>8.4f} {metrics['r2']:>8.4f}"
        )

    # Find best model
    best_model = max(results.keys(), key=lambda k: results[k]["r2"])
    print(f"\nBest Model: {best_model} (R² = {results[best_model]['r2']:.4f})")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"ensemble_{args.task}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    main()
