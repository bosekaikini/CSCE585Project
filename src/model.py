"""Neural networks used by the pricing experiments."""
from __future__ import annotations
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, hidden_layers: int = 3, output_dim: int = 1):
        super().__init__()
        if hidden_layers < 1: raise ValueError("hidden_layers must be positive")
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1): layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class PINN(MLP):
    """Backwards-compatible name for the pricing network."""
    pass
