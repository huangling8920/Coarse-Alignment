from __future__ import annotations

import numpy as np

from src.navigation.rotations import wrap_deg


def rmse(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x**2))) if x.size else float("nan")


def convergence_time(
    times: np.ndarray,
    abs_error_deg: np.ndarray,
    threshold_deg: float,
    hold_s: float,
) -> float:
    times = np.asarray(times, dtype=float)
    err = np.asarray(abs_error_deg, dtype=float)
    if times.size == 0:
        return float("nan")
    for i, t0 in enumerate(times):
        mask = (times >= t0) & (times <= t0 + hold_s)
        if np.any(mask) and times[mask][-1] - t0 >= hold_s and np.all(err[mask] <= threshold_deg):
            return float(t0)
    return float("nan")


def evaluate_heading_series(
    times: np.ndarray,
    heading_est_deg: np.ndarray,
    heading_truth_deg: np.ndarray,
    config: dict,
    valid: np.ndarray | None = None,
) -> dict[str, float]:
    times = np.asarray(times, dtype=float)
    est = np.asarray(heading_est_deg, dtype=float)
    truth = np.asarray(heading_truth_deg, dtype=float)
    if valid is None:
        valid = np.ones_like(times, dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)

    eval_win = config.get("experiment", {}).get("evaluation_window_s", [times[0], times[-1]])
    mask = (times >= float(eval_win[0])) & (times <= float(eval_win[1])) & valid
    err = wrap_deg(est[mask] - truth[mask])
    abs_err = np.abs(err)
    conv = convergence_time(
        times[valid],
        np.abs(wrap_deg(est[valid] - truth[valid])),
        float(config["filter"]["convergence_threshold_deg"]),
        float(config["filter"]["convergence_hold_s"]),
    )
    failed = bool(
        err.size == 0
        or np.nanmax(abs_err) > float(config["filter"]["failure_heading_threshold_deg"])
        or np.isnan(conv)
    )
    return {
        "mean_error_deg": float(np.mean(err)) if err.size else float("nan"),
        "std_error_deg": float(np.std(err)) if err.size else float("nan"),
        "heading_rmse_deg": rmse(err),
        "max_heading_error_deg": float(np.max(abs_err)) if err.size else float("nan"),
        "convergence_time_s": conv,
        "failure": float(failed),
    }


def gyro_bias_rmse_dph(bias_error_dph: np.ndarray) -> float:
    bias_error_dph = np.asarray(bias_error_dph, dtype=float)
    if bias_error_dph.ndim == 2:
        return rmse(np.linalg.norm(bias_error_dph, axis=1))
    return rmse(bias_error_dph)

