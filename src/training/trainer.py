"""
Training loop and utilities for hive prediction models.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Callable, Union, Literal
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

from .metrics import MetricTracker, format_metrics


class EarlyStopping:
    """Early stopping callback to stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics like accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score

        Returns:
            True if should stop training
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class Trainer:
    """
    Training loop for hive prediction models.

    Supports:
    - Classification and regression tasks
    - Sequence and non-sequence models
    - Early stopping
    - Learning rate scheduling
    - Model checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        task_type: Literal["classification", "regression"] = "classification",
        use_sequences: bool = False,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        optimizer: str = "adamw",
        scheduler: Optional[str] = "plateau",
        class_weights: Optional[torch.Tensor] = None,
        device: Optional[str] = None,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            task_type: 'classification' or 'regression'
            use_sequences: Whether model expects sequence inputs
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            optimizer: Optimizer type ('adam' or 'adamw')
            scheduler: LR scheduler type ('plateau', 'cosine', or None)
            class_weights: Optional weights for imbalanced classes
            device: Device to use (auto-detected if None)
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task_type = task_type
        self.use_sequences = use_sequences

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = self.model.to(self.device)

        # Optimizer
        if optimizer == "adamw":
            self.optimizer = AdamW(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.optimizer = Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )

        # Scheduler
        self.scheduler = None
        if scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="min", patience=5, factor=0.5
            )
        elif scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100)

        # Loss function
        if task_type == "classification":
            if class_weights is not None:
                self.criterion = nn.CrossEntropyLoss(
                    weight=class_weights.to(self.device)
                )
            else:
                self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

        # Checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Training state
        self.best_val_loss = float("inf")
        self.history = {"train": [], "val": []}

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        tracker = MetricTracker(self.task_type)

        pbar = tqdm(self.train_loader, desc="Training", leave=False)

        for batch in pbar:
            # Unpack batch
            if self.use_sequences:
                x, y, lengths = batch
                x = x.to(self.device)
                y = y.to(self.device)
                lengths = lengths.to(self.device)
            else:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                lengths = None

            # Forward pass
            self.optimizer.zero_grad()

            if self.use_sequences:
                output, _ = self.model(x, lengths)
            else:
                output, _ = self.model(x)

            # Compute loss
            loss = self.criterion(output, y)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track metrics
            with torch.no_grad():
                if self.task_type == "classification":
                    y_pred = output.argmax(dim=-1).cpu().numpy()
                    y_prob = F.softmax(output, dim=-1).cpu().numpy()
                else:
                    y_pred = output.cpu().numpy()
                    y_prob = None

                tracker.update(
                    y_true=y.cpu().numpy(),
                    y_pred=y_pred,
                    y_prob=y_prob,
                    loss=loss.item(),
                    n_samples=len(y),
                )

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return tracker.compute()

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        tracker = MetricTracker(self.task_type)

        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            if self.use_sequences:
                x, y, lengths = batch
                x = x.to(self.device)
                y = y.to(self.device)
                lengths = lengths.to(self.device)
            else:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                lengths = None

            if self.use_sequences:
                output, _ = self.model(x, lengths)
            else:
                output, _ = self.model(x)

            loss = self.criterion(output, y)

            if self.task_type == "classification":
                y_pred = output.argmax(dim=-1).cpu().numpy()
                y_prob = F.softmax(output, dim=-1).cpu().numpy()
            else:
                y_pred = output.cpu().numpy()
                y_prob = None

            tracker.update(
                y_true=y.cpu().numpy(),
                y_pred=y_pred,
                y_prob=y_prob,
                loss=loss.item(),
                n_samples=len(y),
            )

        return tracker.compute()

    def fit(
        self,
        epochs: int = 100,
        early_stopping_patience: int = 15,
        save_best: bool = True,
        verbose: bool = True,
    ) -> Dict:
        """
        Train the model.

        Args:
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            save_best: Whether to save best model checkpoint
            verbose: Whether to print progress

        Returns:
            Training history
        """
        early_stopping = EarlyStopping(patience=early_stopping_patience, mode="min")

        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch()
            self.history["train"].append(train_metrics)

            # Validate
            val_metrics = self.validate()
            self.history["val"].append(val_metrics)

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # Logging
            if verbose:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"\nEpoch {epoch + 1}/{epochs} (lr={lr:.2e})")
                print(f"  Train: {format_metrics(train_metrics)}")
                print(f"  Val:   {format_metrics(val_metrics)}")

            # Save best model
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                if save_best and self.checkpoint_dir is not None:
                    self.save_checkpoint("best_model.pt")
                    if verbose:
                        print("  -> Saved best model")

            # Early stopping
            if early_stopping(val_metrics["loss"]):
                if verbose:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        return self.history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return

        path = self.checkpoint_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
                "history": self.history,
            },
            path,
        )

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        if self.checkpoint_dir is None:
            return

        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint["history"]


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """
    Compute inverse frequency class weights for imbalanced datasets.

    Args:
        y: Label array

    Returns:
        Tensor of class weights
    """
    classes, counts = np.unique(y, return_counts=True)
    weights = 1.0 / counts
    weights = weights / weights.sum()

    return torch.tensor(weights, dtype=torch.float32)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    task_type: str = "classification",
    use_sequences: bool = False,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    checkpoint_dir: Optional[Path] = None,
    device: Optional[str] = None,
) -> Dict:
    """
    Convenience function to train a model end-to-end.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        task_type: 'classification' or 'regression'
        use_sequences: Whether model expects sequences
        epochs: Number of epochs
        learning_rate: Learning rate
        checkpoint_dir: Checkpoint directory
        device: Device to use

    Returns:
        Dictionary with training history and test results
    """
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task_type=task_type,
        use_sequences=use_sequences,
        learning_rate=learning_rate,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )

    # Train
    history = trainer.fit(epochs=epochs)

    # Load best model and evaluate on test set
    if checkpoint_dir is not None:
        trainer.load_checkpoint("best_model.pt")

    # Test evaluation
    trainer.val_loader = test_loader
    test_metrics = trainer.validate()

    return {"history": history, "test_metrics": test_metrics}


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.data.dataset import create_dataloaders
    from src.models.mlp import create_mlp_model

    print("Testing Trainer with MLP model...")

    # Create dataloaders
    train_loader, val_loader, test_loader, info = create_dataloaders(
        task="population", task_type="classification", batch_size=16
    )

    print(
        f"Train size: {info['train_size']}, Val size: {info['val_size']}, Test size: {info['test_size']}"
    )
    print(f"Num features: {info['num_features']}, Num classes: {info['num_classes']}")

    # Create model
    model = create_mlp_model(
        num_features=info["num_features"],
        hidden_dim=128,
        num_layers=2,
        num_classes=info["num_classes"],
        task_type="classification",
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    checkpoint_dir = Path("checkpoints")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task_type="classification",
        learning_rate=1e-3,
        checkpoint_dir=checkpoint_dir,
    )

    # Train for a few epochs
    print("\nTraining for 5 epochs...")
    history = trainer.fit(epochs=5, verbose=True)

    print("\nDone!")
