#!/usr/bin/env python3
"""
Main training script for hive population prediction models.

Usage:
    uv run python scripts/train.py --model mlp --task population --task-type classification
    uv run python scripts/train.py --model lstm --task population --use-sequences
    uv run python scripts/train.py --model transformer --epochs 50 --batch-size 16
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from src.data.dataset import create_dataloaders
from src.models.mlp import create_mlp_model
from src.models.lstm import create_lstm_model
from src.models.transformer import create_transformer_model
from src.training.trainer import Trainer, train_model
from src.training.metrics import format_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train hive prediction models")

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["mlp", "lstm", "gru", "transformer"],
        help="Model architecture to use",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=128, help="Hidden dimension size"
    )
    parser.add_argument(
        "--num-layers", type=int, default=3, help="Number of model layers"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Task arguments
    parser.add_argument(
        "--task",
        type=str,
        default="population",
        choices=["population", "winter"],
        help="Prediction task",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default="classification",
        choices=["classification", "regression"],
        help="Task type (classification or regression)",
    )

    # Data arguments
    parser.add_argument(
        "--data-path", type=str, default=None, help="Path to MSPB data directory"
    )
    parser.add_argument(
        "--window-days", type=int, default=7, help="Number of days for feature window"
    )
    parser.add_argument(
        "--use-sequences",
        action="store_true",
        help="Use sequence data (for LSTM/Transformer)",
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=100, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Initial learning rate"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--patience", type=int, default=15, help="Early stopping patience"
    )

    # Output arguments
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--experiment-name", type=str, default=None, help="Name for this experiment"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda/cpu)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(args, info):
    """Create model based on arguments."""
    model_type = args.model

    if model_type == "mlp":
        # MLP uses aggregated features
        return create_mlp_model(
            num_features=info["num_features"],
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=info["num_classes"],
            dropout=args.dropout,
            task_type=args.task_type,
        )

    elif model_type in ["lstm", "gru"]:
        return create_lstm_model(
            num_features=info["num_features"],
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=info["num_classes"],
            dropout=args.dropout,
            rnn_type=model_type,
            task_type=args.task_type,
        )

    elif model_type == "transformer":
        return create_transformer_model(
            num_features=info["num_features"],
            d_model=args.hidden_dim,
            num_heads=4,
            num_layers=args.num_layers,
            ff_dim=args.hidden_dim * 4,
            num_classes=info["num_classes"],
            dropout=args.dropout,
            task_type=args.task_type,
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Determine if using sequences
    use_sequences = args.use_sequences or args.model in ["lstm", "gru", "transformer"]

    # For MLP, we use aggregated features (not sequences)
    if args.model == "mlp":
        use_sequences = False

    print("=" * 60)
    print("Hive Population Prediction Training")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Task: {args.task} ({args.task_type})")
    print(f"Use sequences: {use_sequences}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Num layers: {args.num_layers}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print("=" * 60)

    # Create experiment directory
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{args.model}_{args.task}_{timestamp}"

    output_dir = Path(args.output_dir) / args.experiment_name
    checkpoint_dir = output_dir / "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save config
    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nOutput directory: {output_dir}")

    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader, info = create_dataloaders(
        data_path=args.data_path,
        task=args.task,
        task_type=args.task_type,
        batch_size=args.batch_size,
        window_days=args.window_days,
        use_sequences=use_sequences,
    )

    print(f"Train samples: {info['train_size']}")
    print(f"Validation samples: {info['val_size']}")
    print(f"Test samples: {info['test_size']}")
    print(f"Number of features: {info['num_features']}")
    print(f"Number of classes: {info['num_classes']}")

    # Create model
    print("\nCreating model...")
    model = create_model(args, info)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task_type=args.task_type,
        use_sequences=use_sequences,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        checkpoint_dir=checkpoint_dir,
        device=args.device,
    )

    print(f"\nUsing device: {trainer.device}")

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    history = trainer.fit(
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        save_best=True,
        verbose=True,
    )

    # Load best model and evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)

    trainer.load_checkpoint("best_model.pt")
    trainer.val_loader = test_loader
    test_metrics = trainer.validate()

    print(f"\nTest Results: {format_metrics(test_metrics)}")

    # Save results
    results = {
        "config": config,
        "train_history": history["train"],
        "val_history": history["val"],
        "test_metrics": test_metrics,
        "best_val_loss": trainer.best_val_loss,
    }

    with open(output_dir / "results.json", "w") as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump(results, f, indent=2, default=convert)

    print(f"\nResults saved to: {output_dir}")

    # Print final summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Test {args.task_type} metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")

    return results


if __name__ == "__main__":
    main()
