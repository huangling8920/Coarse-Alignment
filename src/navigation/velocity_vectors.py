from __future__ import annotations

import numpy as np


def integrate_trapezoid(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.ndim != 1:
        raise ValueError("times must be 1-D")
    if values.shape[0] != times.shape[0]:
        raise ValueError("values first dimension must match times")
    out = np.zeros_like(values, dtype=float)
    for k in range(1, len(times)):
        dt = float(times[k] - times[k - 1])
        out[k] = out[k - 1] + 0.5 * dt * (values[k] + values[k - 1])
    return out


def gps_velocity_vector(times: np.ndarray, gps_velocity_t: np.ndarray) -> np.ndarray:
    """Construct the GPS-derived velocity-vector observation beta_M."""
    return integrate_trapezoid(times, gps_velocity_t)


def sins_specific_force_vector(times: np.ndarray, specific_force_b: np.ndarray, C_b_t: np.ndarray) -> np.ndarray:
    """Construct the inertial velocity-vector term alpha_M in a simplified form.

    The production manuscript uses a full transverse mechanization. This
    default keeps the same data flow and vector dimensions for reproducibility.
    """
    C_b_t = np.asarray(C_b_t, dtype=float).reshape(3, 3)
    f_t = (C_b_t @ np.asarray(specific_force_b, dtype=float).T).T
    return integrate_trapezoid(times, f_t)

