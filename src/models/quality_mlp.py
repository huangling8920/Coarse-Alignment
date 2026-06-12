from __future__ import annotations

import torch
from torch import nn


class QualityWeightMLP(nn.Module):
    """Lightweight learned visual-inertial quality model.

    Manuscript correspondence: the learned rho_VI model with 8 input
    indicators, 8-16-8-1 MLP topology, ReLU hidden activations, and sigmoid
    output mapped to [rho_min, 1].
    """

    def __init__(self, input_dim: int = 8, hidden_dims: tuple[int, int] | list[int] = (16, 8), rho_min: float = 0.25):
        super().__init__()
        if input_dim != 8:
            raise ValueError("The manuscript quality feature vector has dimension 8")
        self.rho_min = float(rho_min)
        layers: list[nn.Module] = []
        last = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, int(h)))
            layers.append(nn.ReLU())
            last = int(h)
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 8:
            raise ValueError(f"Expected last dimension 8, got {x.shape}")
        raw = torch.sigmoid(self.net(x))
        return self.rho_min + (1.0 - self.rho_min) * raw


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

