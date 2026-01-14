#!/usr/bin/env python3
"""
Evaluation and inference script for hive prediction models.

Usage:
    uv run python scripts/evaluate.py --checkpoint results/experiment/checkpoints/best_model.pt
    uv run python scripts/evaluate.py --checkpoint best_model.pt --config config.json
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from src.data.dataset import create_dataloaders
from src.models.mlp import create_mlp_model
from src.models.lstm import create_lstm_model
from src.models.transformer import create_transformer_model
from src.training.metrics import (
    MetricTracker,
    format_metrics,
    compute_classification_metrics,
    compute_regression_metrics,
    get_confusion_matrix,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate hive prediction models")

    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (auto-detected if in same directory as checkpoint)",
    )
    parser.add_argument(
        "--data-path", type=str, default=None, help="Path to data (overrides config)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument(
        "--output", type=str, default=None, help="Output file for predictions"
    )

    return parser.parse_args()


def load_model_and_config(checkpoint_path: Path, config_path: Path = None):
    """Load model from checkpoint with config."""
    # Try to find config if not provided
    if config_path is None:
        # Look in parent directories
        for parent in [checkpoint_path.parent, checkpoint_path.parent.parent]:
            potential_config = parent / "config.json"
            if potential_config.exists():
                config_path = potential_config
                break

    if config_path is None or not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found. Please provide --config argument."
        )

    with open(config_path) as f:
        config = json.load(f)

    return config


def create_model_from_config(config: dict, info: dict):
    """Create model from config dictionary."""
    model_type = config["model"]

    if model_type == "mlp":
        return create_mlp_model(
            num_features=info["num_features"],
            hidden_dim=config.get("hidden_dim", 128),
            num_layers=config.get("num_layers", 3),
            num_classes=info["num_classes"],
            dropout=config.get("dropout", 0.1),
            task_type=config.get("task_type", "classification"),
        )
    elif model_type in ["lstm", "gru"]:
        return create_lstm_model(
            num_features=info["num_features"],
            hidden_dim=config.get("hidden_dim", 128),
            num_layers=config.get("num_layers", 2),
            num_classes=info["num_classes"],
            dropout=config.get("dropout", 0.1),
            rnn_type=model_type,
            task_type=config.get("task_type", "classification"),
        )
    elif model_type == "transformer":
        return create_transformer_model(
            num_features=info["num_features"],
            d_model=config.get("hidden_dim", 128),
            num_heads=4,
            num_layers=config.get("num_layers", 3),
            ff_dim=config.get("hidden_dim", 128) * 4,
            num_classes=info["num_classes"],
            dropout=config.get("dropout", 0.1),
            task_type=config.get("task_type", "classification"),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate(
    model, dataloader, task_type: str, device: torch.device, use_sequences: bool
):
    """Evaluate model on dataloader."""
    model.eval()

    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    with torch.no_grad():
        for batch in dataloader:
            if use_sequences:
                x, y, lengths = batch
                x = x.to(device)
                y = y.to(device)
                lengths = lengths.to(device)
                output, _ = model(x, lengths)
            else:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                output, _ = model(x)

            if task_type == "classification":
                y_pred = output.argmax(dim=-1).cpu().numpy()
                y_prob = torch.softmax(output, dim=-1).cpu().numpy()
                all_y_prob.append(y_prob)
            else:
                y_pred = output.cpu().numpy()

            all_y_true.append(y.cpu().numpy())
            all_y_pred.append(y_pred)

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    y_prob = np.concatenate(all_y_prob) if all_y_prob else None

    # Compute metrics
    if task_type == "classification":
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        cm_result = get_confusion_matrix(y_true, y_pred, ["Low (<20)", "High (>=20)"])
    else:
        metrics = compute_regression_metrics(y_true, y_pred)
        cm_result = None

    return {
        "metrics": metrics,
        "confusion_matrix": cm_result,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config) if args.config else None

    print("=" * 60)
    print("Hive Model Evaluation")
    print("=" * 60)

    # Load config
    print(f"\nLoading config...")
    config = load_model_and_config(checkpoint_path, config_path)
    print(f"Model: {config['model']}")
    print(f"Task: {config['task']} ({config['task_type']})")

    # Determine if using sequences
    use_sequences = config.get("use_sequences", False) or config["model"] in [
        "lstm",
        "gru",
        "transformer",
    ]
    if config["model"] == "mlp":
        use_sequences = False

    # Create dataloaders
    print("\nLoading data...")
    data_path = args.data_path or config.get("data_path")
    train_loader, val_loader, test_loader, info = create_dataloaders(
        data_path=data_path,
        task=config["task"],
        task_type=config["task_type"],
        batch_size=config.get("batch_size", 32),
        window_days=config.get("window_days", 7),
        use_sequences=use_sequences,
    )

    # Select dataloader based on split
    if args.split == "train":
        dataloader = train_loader
    elif args.split == "val":
        dataloader = val_loader
    else:
        dataloader = test_loader

    print(f"Evaluating on {args.split} split ({len(dataloader.dataset)} samples)")

    # Create model
    print("\nCreating model...")
    model = create_model_from_config(config, info)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Move to device
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(device)
    print(f"Using device: {device}")

    # Evaluate
    print("\nEvaluating...")
    results = evaluate(model, dataloader, config["task_type"], device, use_sequences)

    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"\n{format_metrics(results['metrics'])}")

    if results["confusion_matrix"]:
        print("\nConfusion Matrix:")
        cm = results["confusion_matrix"]["confusion_matrix"]
        print(f"  True\\Pred   Low    High")
        print(f"  Low        {cm[0][0]:5d}  {cm[0][1]:5d}")
        print(f"  High       {cm[1][0]:5d}  {cm[1][1]:5d}")

        print("\nPer-class metrics:")
        for cls, metrics in results["confusion_matrix"]["per_class"].items():
            print(
                f"  {cls}: precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}"
            )

    # Save predictions if requested
    if args.output:
        output_path = Path(args.output)
        np.savez(
            output_path,
            y_true=results["y_true"],
            y_pred=results["y_pred"],
            y_prob=results["y_prob"] if results["y_prob"] is not None else np.array([]),
        )
        print(f"\nPredictions saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
