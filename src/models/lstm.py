"""
LSTM/GRU model for hive prediction using temporal sequences.

This model processes time-series sensor data using recurrent neural networks.
"""

from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Attention-based pooling over sequence dimension.

    Learns to weight different time steps based on their importance
    for the prediction task.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Hidden states of shape (batch_size, seq_len, hidden_dim)
            mask: Optional mask of shape (batch_size, seq_len), 1 for valid, 0 for padding

        Returns:
            Tuple of (pooled_output, attention_weights)
        """
        # Compute attention scores
        scores = self.attention(x).squeeze(-1)  # (batch_size, seq_len)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax over sequence dimension
        weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len)

        # Weighted sum
        output = torch.bmm(weights.unsqueeze(1), x).squeeze(
            1
        )  # (batch_size, hidden_dim)

        return output, weights


class HiveLSTM(nn.Module):
    """
    LSTM/GRU model for hive phenotypic trait prediction.

    Architecture:
    - Input projection layer
    - Bidirectional LSTM/GRU layers
    - Attention-based pooling (or last hidden state)
    - Task-specific output head
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 1,
        dropout: float = 0.1,
        rnn_type: Literal["lstm", "gru"] = "lstm",
        bidirectional: bool = True,
        use_attention: bool = True,
        task_type: str = "regression",
    ):
        """
        Args:
            num_features: Number of input features per time step
            hidden_dim: RNN hidden dimension
            num_layers: Number of RNN layers
            num_classes: Number of output classes (1 for regression)
            dropout: Dropout rate
            rnn_type: 'lstm' or 'gru'
            bidirectional: Whether to use bidirectional RNN
            use_attention: Whether to use attention pooling
            task_type: 'regression' or 'classification'
        """
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.task_type = task_type

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # RNN layers
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Output dimension after RNN
        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Pooling layer
        if use_attention:
            self.pooling = AttentionPooling(rnn_output_dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(rnn_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Task head
        if task_type == "classification":
            self.output_head = nn.Linear(hidden_dim, num_classes)
        else:
            self.output_head = nn.Linear(hidden_dim, 1)

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
            lengths: Optional sequence lengths for masking
            return_attention: Whether to return attention weights

        Returns:
            Tuple of (predictions, attention_weights)
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_proj(x)  # (batch_size, seq_len, hidden_dim)

        # Pack sequences if lengths provided
        if lengths is not None:
            # Sort by length for packing
            lengths_sorted, sort_idx = lengths.sort(descending=True)
            x_sorted = x[sort_idx]

            # Pack
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
            )

            # RNN forward
            rnn_out_packed, _ = self.rnn(x_packed)

            # Unpack
            rnn_out, _ = nn.utils.rnn.pad_packed_sequence(
                rnn_out_packed, batch_first=True
            )

            # Restore original order
            _, unsort_idx = sort_idx.sort()
            rnn_out = rnn_out[unsort_idx]

            # Pad to original length if needed
            if rnn_out.size(1) < seq_len:
                padding = torch.zeros(
                    batch_size,
                    seq_len - rnn_out.size(1),
                    rnn_out.size(2),
                    device=rnn_out.device,
                    dtype=rnn_out.dtype,
                )
                rnn_out = torch.cat([rnn_out, padding], dim=1)
        else:
            rnn_out, _ = self.rnn(x)  # (batch_size, seq_len, hidden_dim * 2)

        # Pooling
        attention_weights = None
        if self.use_attention:
            # Create mask from lengths
            mask = None
            if lengths is not None:
                mask = torch.arange(seq_len, device=x.device).unsqueeze(
                    0
                ) < lengths.unsqueeze(1)

            pooled, attention_weights = self.pooling(rnn_out, mask)
        else:
            # Use last valid hidden state
            if lengths is not None:
                # Gather last valid states
                idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, rnn_out.size(2))
                pooled = rnn_out.gather(1, idx.long()).squeeze(1)
            else:
                pooled = rnn_out[:, -1, :]  # Last time step

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


class DualHeadLSTM(nn.Module):
    """
    LSTM with dual heads for simultaneous regression and classification.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
        rnn_type: Literal["lstm", "gru"] = "lstm",
        bidirectional: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # RNN layers
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        if use_attention:
            self.pooling = AttentionPooling(rnn_output_dim)

        # Shared projection
        self.shared_proj = nn.Sequential(
            nn.Linear(rnn_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Separate heads
        self.regression_head = nn.Linear(hidden_dim, 1)
        self.classification_head = nn.Linear(hidden_dim, num_classes)

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            Tuple of (regression_output, classification_logits)
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_proj(x)

        # RNN
        if lengths is not None:
            lengths_sorted, sort_idx = lengths.sort(descending=True)
            x_sorted = x[sort_idx]
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
            )
            rnn_out_packed, _ = self.rnn(x_packed)
            rnn_out, _ = nn.utils.rnn.pad_packed_sequence(
                rnn_out_packed, batch_first=True
            )
            _, unsort_idx = sort_idx.sort()
            rnn_out = rnn_out[unsort_idx]

            if rnn_out.size(1) < seq_len:
                padding = torch.zeros(
                    batch_size,
                    seq_len - rnn_out.size(1),
                    rnn_out.size(2),
                    device=rnn_out.device,
                    dtype=rnn_out.dtype,
                )
                rnn_out = torch.cat([rnn_out, padding], dim=1)
        else:
            rnn_out, _ = self.rnn(x)

        # Pooling
        if self.use_attention:
            mask = None
            if lengths is not None:
                mask = torch.arange(seq_len, device=x.device).unsqueeze(
                    0
                ) < lengths.unsqueeze(1)
            pooled, _ = self.pooling(rnn_out, mask)
        else:
            if lengths is not None:
                idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, rnn_out.size(2))
                pooled = rnn_out.gather(1, idx.long()).squeeze(1)
            else:
                pooled = rnn_out[:, -1, :]

        # Shared projection
        h = self.shared_proj(pooled)

        # Heads
        reg_out = self.regression_head(h).squeeze(-1)
        cls_out = self.classification_head(h)

        return reg_out, cls_out


def create_lstm_model(
    num_features: int,
    hidden_dim: int = 128,
    num_layers: int = 2,
    num_classes: int = 2,
    dropout: float = 0.1,
    rnn_type: str = "lstm",
    bidirectional: bool = True,
    use_attention: bool = True,
    task_type: str = "classification",
    dual_head: bool = False,
) -> nn.Module:
    """
    Factory function to create an LSTM/GRU model.
    """
    if dual_head:
        return DualHeadLSTM(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            use_attention=use_attention,
        )
    else:
        return HiveLSTM(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            use_attention=use_attention,
            task_type=task_type,
        )


if __name__ == "__main__":
    # Test the models
    batch_size = 16
    seq_len = 100
    num_features = 22

    print("Testing HiveLSTM (classification)...")
    model = HiveLSTM(num_features, num_classes=2, task_type="classification")
    x = torch.randn(batch_size, seq_len, num_features)
    lengths = torch.randint(50, seq_len, (batch_size,))
    out, attn = model(x, lengths, return_attention=True)
    print(
        f"Input: {x.shape}, Output: {out.shape}, Attention: {attn.shape if attn is not None else None}"
    )

    print("\nTesting HiveLSTM (regression)...")
    model = HiveLSTM(num_features, num_classes=1, task_type="regression")
    out, attn = model(x, lengths)
    print(f"Input: {x.shape}, Output: {out.shape}")

    print("\nTesting DualHeadLSTM...")
    model = DualHeadLSTM(num_features)
    reg_out, cls_out = model(x, lengths)
    print(
        f"Input: {x.shape}, Regression: {reg_out.shape}, Classification: {cls_out.shape}"
    )

    print("\nTesting GRU variant...")
    model = HiveLSTM(num_features, rnn_type="gru", task_type="classification")
    out, _ = model(x, lengths)
    print(f"GRU Output: {out.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
