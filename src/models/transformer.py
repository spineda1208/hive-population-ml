"""
Transformer model for hive prediction using temporal sequences.

This model uses self-attention to capture temporal dependencies and
cross-modal interactions in the sensor data.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding as an alternative to sinusoidal.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention layer.
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply mask
        if mask is not None:
            # mask: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, v)

        # Reshape and project
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )
        out = self.out_proj(context)

        if return_attention:
            return out, attn_weights.mean(dim=1)  # Average over heads
        return out, None


class TransformerBlock(nn.Module):
    """
    Single transformer encoder block.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-attention with residual
        attn_out, attn_weights = self.attention(x, mask, return_attention)
        x = self.norm1(x + attn_out)

        # Feedforward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x, attn_weights


class HiveTransformer(nn.Module):
    """
    Transformer model for hive phenotypic trait prediction.

    Architecture:
    - Input embedding layer
    - Positional encoding
    - Stack of transformer encoder blocks
    - Global pooling (CLS token or mean pooling)
    - Task-specific output head
    """

    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        ff_dim: int = 512,
        num_classes: int = 1,
        dropout: float = 0.1,
        max_seq_len: int = 2016,
        use_cls_token: bool = True,
        task_type: str = "regression",
    ):
        """
        Args:
            num_features: Number of input features per time step
            d_model: Transformer hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            ff_dim: Feedforward network dimension
            num_classes: Number of output classes (1 for regression)
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            use_cls_token: Whether to use CLS token for pooling
            task_type: 'regression' or 'classification'
        """
        super().__init__()

        self.num_features = num_features
        self.d_model = d_model
        self.use_cls_token = use_cls_token
        self.task_type = task_type

        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(num_features, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout)
        )

        # CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len + 1, dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Task head
        if task_type == "classification":
            self.output_head = nn.Linear(d_model // 2, num_classes)
        else:
            self.output_head = nn.Linear(d_model // 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
            lengths: Optional sequence lengths
            return_attention: Whether to return attention weights

        Returns:
            Tuple of (predictions, attention_weights)
        """
        batch_size, seq_len, _ = x.shape

        # Input embedding
        x = self.input_embed(x)  # (batch_size, seq_len, d_model)

        # Prepend CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, seq_len + 1, d_model)

        # Positional encoding
        x = self.pos_encoding(x)

        # Create attention mask
        mask = None
        if lengths is not None:
            # Create mask for valid positions
            if self.use_cls_token:
                mask = torch.arange(seq_len + 1, device=x.device).unsqueeze(
                    0
                ) <= lengths.unsqueeze(1)
            else:
                mask = torch.arange(seq_len, device=x.device).unsqueeze(
                    0
                ) < lengths.unsqueeze(1)

        # Transformer blocks
        attention_weights = None
        for block in self.blocks:
            x, attn = block(x, mask, return_attention)
            if return_attention:
                attention_weights = attn

        # Pooling
        if self.use_cls_token:
            pooled = x[:, 0, :]  # CLS token representation
        else:
            # Mean pooling over valid positions
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(
                    dim=1
                ).clamp(min=1)
            else:
                pooled = x.mean(dim=1)

        # Output projection
        h = self.output_proj(pooled)

        # Task head
        out = self.output_head(h)

        if self.task_type == "regression":
            out = out.squeeze(-1)

        if return_attention:
            return out, attention_weights
        return out, None

    def predict(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get predictions (probabilities for classification)."""
        out, _ = self.forward(x, lengths)

        if self.task_type == "classification":
            return F.softmax(out, dim=-1)
        return out


class DualHeadTransformer(nn.Module):
    """
    Transformer with dual heads for simultaneous regression and classification.
    """

    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        ff_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 2016,
        use_cls_token: bool = True,
    ):
        super().__init__()

        self.num_features = num_features
        self.d_model = d_model
        self.use_cls_token = use_cls_token

        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(num_features, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout)
        )

        # CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len + 1, dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        # Shared output projection
        self.shared_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Separate heads
        self.regression_head = nn.Linear(d_model // 2, 1)
        self.classification_head = nn.Linear(d_model // 2, num_classes)

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            Tuple of (regression_output, classification_logits)
        """
        batch_size, seq_len, _ = x.shape

        # Input embedding
        x = self.input_embed(x)

        # Prepend CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        # Positional encoding
        x = self.pos_encoding(x)

        # Create mask
        mask = None
        if lengths is not None:
            if self.use_cls_token:
                mask = torch.arange(seq_len + 1, device=x.device).unsqueeze(
                    0
                ) <= lengths.unsqueeze(1)
            else:
                mask = torch.arange(seq_len, device=x.device).unsqueeze(
                    0
                ) < lengths.unsqueeze(1)

        # Transformer blocks
        for block in self.blocks:
            x, _ = block(x, mask)

        # Pooling
        if self.use_cls_token:
            pooled = x[:, 0, :]
        else:
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(
                    dim=1
                ).clamp(min=1)
            else:
                pooled = x.mean(dim=1)

        # Shared projection
        h = self.shared_proj(pooled)

        # Separate heads
        reg_out = self.regression_head(h).squeeze(-1)
        cls_out = self.classification_head(h)

        return reg_out, cls_out


def create_transformer_model(
    num_features: int,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 3,
    ff_dim: int = 512,
    num_classes: int = 2,
    dropout: float = 0.1,
    max_seq_len: int = 2016,
    task_type: str = "classification",
    dual_head: bool = False,
) -> nn.Module:
    """
    Factory function to create a Transformer model.
    """
    if dual_head:
        return DualHeadTransformer(
            num_features=num_features,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            num_classes=num_classes,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
    else:
        return HiveTransformer(
            num_features=num_features,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            num_classes=num_classes,
            dropout=dropout,
            max_seq_len=max_seq_len,
            task_type=task_type,
        )


if __name__ == "__main__":
    # Test the models
    batch_size = 16
    seq_len = 100
    num_features = 22

    print("Testing HiveTransformer (classification)...")
    model = HiveTransformer(num_features, num_classes=2, task_type="classification")
    x = torch.randn(batch_size, seq_len, num_features)
    lengths = torch.randint(50, seq_len, (batch_size,))
    out, attn = model(x, lengths, return_attention=True)
    print(
        f"Input: {x.shape}, Output: {out.shape}, Attention: {attn.shape if attn is not None else None}"
    )

    print("\nTesting HiveTransformer (regression)...")
    model = HiveTransformer(num_features, num_classes=1, task_type="regression")
    out, _ = model(x, lengths)
    print(f"Input: {x.shape}, Output: {out.shape}")

    print("\nTesting DualHeadTransformer...")
    model = DualHeadTransformer(num_features)
    reg_out, cls_out = model(x, lengths)
    print(
        f"Input: {x.shape}, Regression: {reg_out.shape}, Classification: {cls_out.shape}"
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
