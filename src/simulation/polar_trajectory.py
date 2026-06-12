from __future__ import annotations

import numpy as np
import pandas as pd


def generate_reference_truth(duration_s: float, gps_rate_hz: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_s + 1e-9, 1.0 / gps_rate_hz)
    roll = 1.2 * np.sin(0.018 * t) + rng.normal(0.0, 0.01, size=t.shape)
    pitch = 0.8 * np.cos(0.015 * t + 0.2) + rng.normal(0.0, 0.01, size=t.shape)
    heading = 60.0 + 1.5 * np.sin(0.011 * t) + 0.25 * np.cos(0.047 * t)
    bgx = 0.08 + 0.012 * np.sin(0.010 * t)
    bgy = -0.05 + 0.010 * np.cos(0.012 * t)
    bgz = 0.12 + 0.018 * np.sin(0.009 * t + 0.5)
    return pd.DataFrame(
        {
            "t": t,
            "roll_deg": roll,
            "pitch_deg": pitch,
            "heading_deg": heading,
            "bgx_dph": bgx,
            "bgy_dph": bgy,
            "bgz_dph": bgz,
        }
    )


def generate_velocity(times: np.ndarray, speed_mps: float = 15.0) -> np.ndarray:
    t = np.asarray(times, dtype=float)
    vx = speed_mps + 0.8 * np.sin(0.021 * t)
    vy = 0.6 * np.cos(0.017 * t)
    vz = 0.04 * np.sin(0.011 * t)
    return np.vstack([vx, vy, vz]).T

