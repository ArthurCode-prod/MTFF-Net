"""Neural network modules for joint crack and rebar-stress forecasting."""

from __future__ import annotations

import torch
from torch import nn


class AttentionPool(nn.Module):
    """Learned temporal pooling over encoder states."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = max(dim // 2, 1)
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(weights * x, dim=1)


class GRUBaseline(nn.Module):
    """Compact recurrent baseline with a single multi-output head."""

    def __init__(self, n_features: int, n_outputs: int, hidden_size: int = 64, dropout: float = 0.15) -> None:
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1])


class MTTFNet(nn.Module):
    """Multi-task temporal fusion network.

    The encoder is shared by all monitoring channels. Crack and rebar-stress
    residual heads are separated after attention pooling so each task can keep
    its own output geometry while still benefiting from common temporal states.
    """

    def __init__(
        self,
        n_features: int,
        n_crack: int,
        n_rebar: int,
        target_indices: list[int],
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.15,
        max_len: int = 128,
        residual_forecast: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_crack = n_crack
        self.n_rebar = n_rebar
        self.target_indices = target_indices
        self.residual_forecast = residual_forecast

        self.input_projection = nn.Linear(n_features, d_model)
        self.position = nn.Parameter(torch.zeros(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.local_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
        )
        self.pool = AttentionPool(d_model)
        self.shared = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.crack_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_crack),
        )
        self.rebar_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_rebar),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.position.shape[1]:
            raise ValueError("Input sequence length exceeds max_len used to initialize the model.")
        z = self.input_projection(x)
        z = z + self.position[:, : z.shape[1]]
        z = self.encoder(z)
        local = self.local_conv(z.transpose(1, 2)).transpose(1, 2)
        h = self.shared(self.pool(z + local))
        residual = torch.cat([self.crack_head(h), self.rebar_head(h)], dim=1)
        if not self.residual_forecast:
            return residual
        last_observed = x[:, -1, self.target_indices]
        return last_observed + residual
