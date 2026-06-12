from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.navigation.rotations import quat_wxyz_to_rot, skew, so3_log


@dataclass(frozen=True)
class CameraIMUCalibration:
    C_c_b: np.ndarray
    p_c_b: np.ndarray
    time_offset_s: float = 0.0
    monocular_scale: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "C_c_b", np.asarray(self.C_c_b, dtype=float).reshape(3, 3))
        object.__setattr__(self, "p_c_b", np.asarray(self.p_c_b, dtype=float).reshape(3))


@dataclass(frozen=True)
class VIMotion:
    delta_p_c: np.ndarray
    delta_v_c: np.ndarray
    delta_q_c_wxyz: np.ndarray
    omega_i_b: np.ndarray
    omega_j_b: np.ndarray


def map_vi_to_transverse(
    motion: VIMotion,
    C_hat_b_t_i: np.ndarray,
    calib: CameraIMUCalibration,
) -> np.ndarray:
    """Map VI relative motion to a 6-D transverse auxiliary observation.

    Corresponds to the manuscript derivation that forms y_M^VI from
    short-window velocity and attitude increments. The output is
    [Delta v_t, Delta theta_t] in R^6.
    """
    C_hat_b_t_i = np.asarray(C_hat_b_t_i, dtype=float).reshape(3, 3)
    dv_c = np.asarray(motion.delta_v_c, dtype=float).reshape(3)
    omega_i = np.asarray(motion.omega_i_b, dtype=float).reshape(3)
    omega_j = np.asarray(motion.omega_j_b, dtype=float).reshape(3)

    dv_b = calib.C_c_b @ (calib.monocular_scale * dv_c)
    lever = (skew(omega_j) - skew(omega_i)) @ calib.p_c_b
    delta_v_t = C_hat_b_t_i @ (dv_b - lever)

    delta_C_c = quat_wxyz_to_rot(motion.delta_q_c_wxyz)
    delta_C_b = calib.C_c_b @ delta_C_c @ calib.C_c_b.T
    delta_theta_t = so3_log(C_hat_b_t_i @ delta_C_b @ C_hat_b_t_i.T)

    y_vi = np.concatenate([delta_v_t, delta_theta_t])
    if y_vi.shape != (6,):
        raise ValueError("VI auxiliary measurement must have dimension 6")
    return y_vi


def base_vi_covariance(vi_velocity_std: float, vi_attitude_std_rad: float) -> np.ndarray:
    R = np.diag([vi_velocity_std**2] * 3 + [vi_attitude_std_rad**2] * 3)
    return R.astype(float)


def scale_vi_covariance(R_vi: np.ndarray, rho_vi: float, rho_min: float = 0.25) -> np.ndarray:
    rho = float(np.clip(rho_vi, rho_min, 1.0))
    return np.asarray(R_vi, dtype=float).reshape(6, 6) / rho

