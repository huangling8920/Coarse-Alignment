from __future__ import annotations

import numpy as np

from src.evaluators.metrics import evaluate_heading_series, rmse


def test_heading_metrics():
    cfg = {
        "experiment": {"evaluation_window_s": [0.0, 4.0]},
        "filter": {"convergence_threshold_deg": 0.5, "convergence_hold_s": 2.0, "failure_heading_threshold_deg": 2.0},
    }
    t = np.arange(5.0)
    truth = np.zeros_like(t)
    est = np.array([1.0, 0.4, 0.3, 0.2, 0.1])
    metrics = evaluate_heading_series(t, est, truth, cfg)
    assert np.isclose(metrics["heading_rmse_deg"], rmse(est))
    assert metrics["failure"] == 0.0
