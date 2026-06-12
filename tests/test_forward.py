from __future__ import annotations

import torch

from src.models.quality_mlp import QualityWeightMLP


def test_quality_model_forward_shape_and_range():
    model = QualityWeightMLP()
    x = torch.rand(4, 8)
    y = model(x)
    assert y.shape == (4, 1)
    assert torch.all(y >= 0.25)
    assert torch.all(y <= 1.0)

