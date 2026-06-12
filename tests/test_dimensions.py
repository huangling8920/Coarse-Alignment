from __future__ import annotations

import numpy as np

from src.navigation.kalman import ClosedLoopAlignmentKalman, KalmanConfig, make_stacked_measurement
from src.vio.vi_measurement import base_vi_covariance, scale_vi_covariance


def test_kalman_stacked_measurement_dimensions():
    kf = ClosedLoopAlignmentKalman(
        KalmanConfig(
            process_attitude_std_rad=1e-3,
            process_bias_std_rad_s=1e-6,
            initial_attitude_error_rad=np.zeros(3),
            initial_bias_error_rad_s=np.zeros(3),
        )
    )
    R_gps = np.eye(3) * 0.1
    R_vi = scale_vi_covariance(base_vi_covariance(0.1, 0.01), 0.5)
    residual, H, R = make_stacked_measurement(np.zeros(3), np.zeros(6), R_gps, R_vi)
    assert residual.shape == (9,)
    assert H.shape == (9, 6)
    assert R.shape == (9, 9)
    kf.predict(1.0)
    innovation = kf.update(residual, H, R)
    assert innovation.shape == (9,)

