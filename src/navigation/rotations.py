from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def skew(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=float,
    )


def so3_exp(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float).reshape(3)
    return Rotation.from_rotvec(phi).as_matrix()


def so3_log(C: np.ndarray) -> np.ndarray:
    C = np.asarray(C, dtype=float).reshape(3, 3)
    return Rotation.from_matrix(C).as_rotvec()


def euler_deg_to_rot(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    return Rotation.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()


def rot_to_euler_deg(C: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(C).reshape(3, 3)).as_euler("xyz", degrees=True)


def quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    q = q / np.linalg.norm(q)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def rot_to_quat_wxyz(C: np.ndarray) -> np.ndarray:
    q_xyzw = Rotation.from_matrix(np.asarray(C).reshape(3, 3)).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)


def wrap_deg(angle_deg: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle_deg) + 180.0) % 360.0 - 180.0


def assert_rot(C: np.ndarray, name: str = "C") -> None:
    C = np.asarray(C, dtype=float)
    if C.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {C.shape}")
    err = np.linalg.norm(C @ C.T - np.eye(3))
    if err > 1e-5:
        raise ValueError(f"{name} is not a valid rotation matrix; orthogonality error={err:.3e}")

