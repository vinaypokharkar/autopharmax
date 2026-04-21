"""Shared encoders for cell-line omics and drug molecular features.

Both encoders are dense BN-ReLU-Dropout stacks. The key invariant for this
project is that a SINGLE DrugEncoder instance is reused by both the
single-drug head and the synergy head (weight sharing happens by object
identity, not by config identity).
"""
from __future__ import annotations

import torch.nn as nn


def _mlp(input_dim: int, hidden_dims: list[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    return nn.Sequential(*layers)


class CellLineEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.3):
        super().__init__()
        self.net = _mlp(input_dim, hidden_dims, dropout)
        self.output_dim = hidden_dims[-1]

    def forward(self, x):
        return self.net(x)


class DrugEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.3):
        super().__init__()
        self.net = _mlp(input_dim, hidden_dims, dropout)
        self.output_dim = hidden_dims[-1]

    def forward(self, x):
        return self.net(x)


class RegressionHead(nn.Module):
    """Dense regression head. Final layer has no activation (real-valued output)."""

    def __init__(self, input_dim: int, head_dims: list[int], dropout: float = 0.2):
        super().__init__()
        assert head_dims[-1] == 1, "regression head must end in 1-dim output"
        layers: list[nn.Module] = []
        prev = input_dim
        for i, h in enumerate(head_dims):
            layers.append(nn.Linear(prev, h))
            if i < len(head_dims) - 1:
                layers += [nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
