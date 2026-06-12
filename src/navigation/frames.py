from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rotations import assert_rot, euler_deg_to_rot


@dataclass(frozen=True)
class TransverseFrameConfig:
    latitude_deg: float = 85.0
    longitude_deg: float = 118.0


def transverse_rotation_from_geodetic(latitude_deg: float, longitude_deg: float) -> np.ndarray:
    """Approximate transverse-frame rotation.

    paper not specified: the manuscript uses the transverse frame but does not
    provide every implementation detail needed for executable software. This
    default keeps a deterministic polar-compatible frame and can be replaced
    by a project-specific geodetic library.
    """
    C_lon = euler_deg_to_rot(0.0, 0.0, longitude_deg)
    C_lat = euler_deg_to_rot(0.0, 90.0 - latitude_deg, 0.0)
    C = C_lat @ C_lon
    assert_rot(C, "C_transverse")
    return C


def body_to_transverse(C_b_n: np.ndarray, frame: TransverseFrameConfig) -> np.ndarray:
    C_t_n = transverse_rotation_from_geodetic(frame.latitude_deg, frame.longitude_deg)
    C_b_t = C_t_n @ np.asarray(C_b_n, dtype=float).reshape(3, 3)
    assert_rot(C_b_t, "C_b_t")
    return C_b_t

