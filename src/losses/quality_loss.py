from __future__ import annotations

import torch


def quality_regression_loss(pred_rho: torch.Tensor, target_rho: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.smooth_l1_loss(pred_rho.reshape_as(target_rho), target_rho)


def innovation_nll_proxy_loss(
    pred_rho: torch.Tensor,
    target_rho: torch.Tensor,
    heading_proxy: torch.Tensor | None = None,
    heading_proxy_weight: float = 0.05,
) -> torch.Tensor:
    """Replaceable default training objective.

    paper not specified: the manuscript states that the model is trained with
    innovation negative log-likelihood and heading-error proxy regularization,
    but does not give every target-construction detail. This loss keeps the
    stated structure and can be swapped without changing the pipeline.
    """
    loss = quality_regression_loss(pred_rho, target_rho)
    if heading_proxy is not None:
        uncertainty = 1.0 / torch.clamp(pred_rho, min=1e-4)
        loss = loss + heading_proxy_weight * torch.mean(uncertainty.reshape_as(heading_proxy) * heading_proxy)
    return loss

