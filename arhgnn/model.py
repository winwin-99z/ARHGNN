from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HypergraphConv(nn.Module):
    """Hypergraph convolution over precomputed edge-type propagation matrices."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_type_count: int,
        dropout: float = 0.5,
        use_attention: bool = True,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_channels)

        if use_attention:
            self.type_embeddings = nn.Parameter(torch.empty(edge_type_count, out_channels))
            self.attention_vector = nn.Parameter(torch.empty(out_channels))
            nn.init.xavier_uniform_(self.type_embeddings)
            nn.init.normal_(self.attention_vector, std=0.02)
        else:
            self.register_parameter("type_embeddings", None)
            self.register_parameter("attention_vector", None)

        if use_residual:
            self.residual = nn.Identity() if in_channels == out_channels else nn.Linear(in_channels, out_channels, bias=False)
        else:
            self.residual = None

    def forward(self, x: torch.Tensor, propagation: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        messages = torch.einsum("tnm,mc->tnc", propagation, h)

        if self.use_attention:
            typed_messages = torch.tanh(messages + self.type_embeddings[:, None, :])
            scores = torch.einsum("tnc,c->nt", typed_messages, self.attention_vector)
            alpha = torch.softmax(scores, dim=1)
            out = torch.einsum("nt,tnc->nc", alpha, messages)
        else:
            out = messages.mean(dim=0)

        if self.residual is not None:
            out = out + self.residual(x)
        return self.dropout(F.relu(self.norm(out)))


class ARHGNN(nn.Module):
    """Attention-residual hypergraph neural network for multi-label prediction."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        edge_type_count: int,
        dropout: float = 0.5,
        use_attention: bool = True,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = HypergraphConv(
            in_channels,
            hidden_channels,
            edge_type_count,
            dropout=dropout,
            use_attention=use_attention,
            use_residual=use_residual,
        )
        self.conv2 = HypergraphConv(
            hidden_channels,
            hidden_channels,
            edge_type_count,
            dropout=dropout,
            use_attention=use_attention,
            use_residual=use_residual,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x: torch.Tensor, propagation: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, propagation)
        h = self.conv2(h, propagation)
        return self.classifier(h)


class FocalLoss(nn.Module):
    """Focal loss for separate sensitivity experiments only."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt).pow(self.gamma) * bce).mean()

