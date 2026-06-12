from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanConfig:
    process_attitude_std_rad: float
    process_bias_std_rad_s: float
    initial_attitude_error_rad: np.ndarray
    initial_bias_error_rad_s: np.ndarray
    initial_covariance: float = 1.0


class ClosedLoopAlignmentKalman:
    """6-state closed-loop Kalman filter used by the proposed method.

    State: x = [delta_theta_x, delta_theta_y, delta_theta_z,
                epsilon_x, epsilon_y, epsilon_z]^T in R^6.
    Measurements are stacked as [GPS attitude-like residual, VI velocity
    residual, VI attitude residual]^T in R^9.
    """

    state_dim = 6

    def __init__(self, cfg: KalmanConfig):
        self.x = np.zeros(6, dtype=float)
        self.x[:3] = np.asarray(cfg.initial_attitude_error_rad, dtype=float).reshape(3)
        self.x[3:] = np.asarray(cfg.initial_bias_error_rad_s, dtype=float).reshape(3)
        self.P = np.eye(6, dtype=float) * float(cfg.initial_covariance)
        self.q_att = float(cfg.process_attitude_std_rad) ** 2
        self.q_bias = float(cfg.process_bias_std_rad_s) ** 2

    def predict(self, dt: float) -> None:
        F = np.eye(6, dtype=float)
        F[:3, 3:] = -np.eye(3) * dt
        Q = np.diag([self.q_att * dt] * 3 + [self.q_bias * dt] * 3)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, residual: np.ndarray, H: np.ndarray, R: np.ndarray) -> np.ndarray:
        residual = np.asarray(residual, dtype=float).reshape(-1)
        H = np.asarray(H, dtype=float)
        R = np.asarray(R, dtype=float)
        if H.shape[1] != 6:
            raise ValueError(f"H must have 6 columns, got {H.shape}")
        if residual.shape[0] != H.shape[0]:
            raise ValueError("residual dimension must match H rows")
        if R.shape != (H.shape[0], H.shape[0]):
            raise ValueError(f"R must be {(H.shape[0], H.shape[0])}, got {R.shape}")
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        innovation = residual - H @ self.x
        self.x = self.x + K @ innovation
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return innovation

    @property
    def attitude_error_rad(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def bias_error_rad_s(self) -> np.ndarray:
        return self.x[3:].copy()


def make_stacked_measurement(
    gps_residual: np.ndarray,
    vi_residual: np.ndarray | None,
    R_gps: np.ndarray,
    R_vi_scaled: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gps_residual = np.asarray(gps_residual, dtype=float).reshape(3)
    H_gps = np.hstack([np.eye(3), np.zeros((3, 3))])
    if vi_residual is None:
        return gps_residual, H_gps, np.asarray(R_gps, dtype=float).reshape(3, 3)

    vi_residual = np.asarray(vi_residual, dtype=float).reshape(6)
    H_vi_vel = np.hstack([0.05 * np.eye(3), np.zeros((3, 3))])
    H_vi_att = np.hstack([np.eye(3), np.zeros((3, 3))])
    H = np.vstack([H_gps, H_vi_vel, H_vi_att])
    residual = np.concatenate([gps_residual, vi_residual])
    R = np.zeros((9, 9), dtype=float)
    R[:3, :3] = np.asarray(R_gps, dtype=float).reshape(3, 3)
    R[3:, 3:] = np.asarray(R_vi_scaled, dtype=float).reshape(6, 6)
    return residual, H, R
