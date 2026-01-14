"""
MLP (Multi-Layer Perceptron) baseline model for hive prediction.

This model operates on aggregated statistical features extracted from
sensor data windows.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    """A single MLP block with linear layer, normalization, activation, and dropout."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class HiveMLP(nn.Module):
    """
    MLP model for hive phenotypic trait prediction.

    Architecture:
    - Input projection
    - Multiple MLP blocks with residual connections
    - Task-specific output head(s)

    Supports both regression and classification tasks.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_classes: int = 1,
        dropout: float = 0.1,
        task_type: str = "regression",
        activation: str = "relu",
    ):
        """
        Args:
            num_features: Number of input features
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP blocks
            num_classes: Number of output classes (1 for regression)
            dropout: Dropout rate
            task_type: 'regression' or 'classification'
            activation: Activation function ('relu', 'gelu', 'silu')
        """
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.task_type = task_type

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # MLP blocks
        self.blocks = nn.ModuleList(
            [
                MLPBlock(hidden_dim, hidden_dim, dropout, activation)
                for _ in range(num_layers)
            ]
        )

        # Output head
        if task_type == "classification":
            self.output_head = nn.Linear(hidden_dim, num_classes)
        else:
            # Regression head with a small MLP
            self.output_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, num_features)
            return_features: Whether to return intermediate features

        Returns:
            Tuple of (predictions, features) where features is None if not requested
        """
        # Input projection
        h = self.input_proj(x)

        # MLP blocks with residual connections
        for block in self.blocks:
            h = h + block(h)

        # Output head
        out = self.output_head(h)

        if self.task_type == "regression":
            out = out.squeeze(-1)  # (batch_size,)

        if return_features:
            return out, h
        return out, None

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predictions (probabilities for classification)."""
        out, _ = self.forward(x)

        if self.task_type == "classification":
            return F.softmax(out, dim=-1)
        return out


class DualHeadMLP(nn.Module):
    """
    MLP with dual heads for simultaneous regression and classification.

    Useful for population prediction where we want both:
    - Exact frame count (regression)
    - High/low classification
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim

        # Shared backbone
        self.input_proj = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList(
            [
                MLPBlock(hidden_dim, hidden_dim, dropout, activation)
                for _ in range(num_layers)
            ]
        )

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Classification head
        self.classification_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, num_features)

        Returns:
            Tuple of (regression_output, classification_logits)
        """
        # Shared backbone
        h = self.input_proj(x)
        for block in self.blocks:
            h = h + block(h)

        # Separate heads
        reg_out = self.regression_head(h).squeeze(-1)
        cls_out = self.classification_head(h)

        return reg_out, cls_out


def create_mlp_model(
    num_features: int,
    hidden_dim: int = 256,
    num_layers: int = 3,
    num_classes: int = 2,
    dropout: float = 0.1,
    task_type: str = "classification",
    dual_head: bool = False,
) -> nn.Module:
    """
    Factory function to create an MLP model.

    Args:
        num_features: Number of input features
        hidden_dim: Hidden layer dimension
        num_layers: Number of MLP blocks
        num_classes: Number of output classes
        dropout: Dropout rate
        task_type: 'regression' or 'classification'
        dual_head: Whether to use dual-head model

    Returns:
        MLP model instance
    """
    if dual_head:
        return DualHeadMLP(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        )
    else:
        return HiveMLP(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            task_type=task_type,
        )


if __name__ == "__main__":
    # Test the models
    batch_size = 16
    num_features = 243

    print("Testing HiveMLP (classification)...")
    model = HiveMLP(num_features, num_classes=2, task_type="classification")
    x = torch.randn(batch_size, num_features)
    out, features = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}")

    print("\nTesting HiveMLP (regression)...")
    model = HiveMLP(num_features, num_classes=1, task_type="regression")
    out, features = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}")

    print("\nTesting DualHeadMLP...")
    model = DualHeadMLP(num_features)
    reg_out, cls_out = model(x)
    print(
        f"Input: {x.shape}, Regression: {reg_out.shape}, Classification: {cls_out.shape}"
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
