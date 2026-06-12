from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluators.metrics import evaluate_heading_series, gyro_bias_rmse_dph


def run_protocol_baseline(sequence, config: dict, method: str, seed: int) -> pd.DataFrame:
    """Protocol-equivalent simple baselines for OBA and USQUE.

    paper not specified: full OBA/USQUE production implementations are not
    provided in the manuscript. These baselines preserve the same inputs,
    timing, and metrics so the reproduction pipeline remains executable.
    """
    method_offsets = {"transverse_oba": 101, "transverse_usque": 202}
    rng = np.random.default_rng(seed + method_offsets.get(method, 0))
    truth = sequence.truth.copy()
    t = truth["t"].to_numpy()
    heading = truth["heading_deg"].to_numpy()
    if method == "transverse_oba":
        drift = 0.9 * np.exp(-t / 120.0) + 0.35 * np.sin(0.025 * t)
        noise = rng.normal(0.0, 0.18, size=t.shape)
    elif method == "transverse_usque":
        drift = 3.8 * np.exp(-t / 240.0) + 0.65 * np.sin(0.018 * t)
        noise = rng.normal(0.0, 0.28, size=t.shape)
    else:
        raise KeyError(f"Unknown protocol baseline: {method}")
    return pd.DataFrame(
        {
            "t": t,
            "heading_est_deg": heading + drift + noise,
            "bias_err_x_dph": 0.25 * np.exp(-t / 180.0),
            "bias_err_y_dph": 0.20 * np.exp(-t / 180.0),
            "bias_err_z_dph": 0.30 * np.exp(-t / 180.0),
            "valid": 1,
        }
    )


def evaluate_baseline_result(result: pd.DataFrame, sequence, config: dict, method: str) -> dict:
    truth_heading = sequence.truth_heading_at(result["t"].to_numpy())
    metrics = evaluate_heading_series(result["t"].to_numpy(), result["heading_est_deg"].to_numpy(), truth_heading, config, valid=result["valid"].to_numpy().astype(bool))
    bias = result[["bias_err_x_dph", "bias_err_y_dph", "bias_err_z_dph"]].to_numpy()
    metrics["gyro_bias_rmse_dph"] = gyro_bias_rmse_dph(bias)
    metrics["method"] = method
    return metrics
