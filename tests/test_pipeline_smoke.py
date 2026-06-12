from __future__ import annotations

from pathlib import Path

from src.experiments.runner import evaluate_dataset
from src.simulation.sensor_simulator import generate_toy_dataset


def test_toy_pipeline_smoke(tmp_path: Path):
    cfg = {
        "project": {"name": "smoke", "seed": 1},
        "runtime": {"device": "cpu"},
        "data": {
            "duration_s": 8.0,
            "imu_rate_hz": 20,
            "camera_rate_hz": 5,
            "gps_rate_hz": 1,
            "image_width": 64,
            "image_height": 48,
            "n_ref_features": 200,
            "seeds": [1],
        },
        "sensor": {
            "gyro_bias_dph": 1.0,
            "normal_gps_sigma_v": 0.03,
            "degraded_gps_sigma_v": 0.30,
            "degraded_gps_window_s": [3.0, 5.0],
            "gps_outages_s": [[4.0, 4.0]],
        },
        "calibration": {
            "C_c_b": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "p_c_b": [0.0, 0.0, 0.0],
            "time_offset_s": 0.0,
            "monocular_scale": 1.0,
        },
        "quality_model": {"input_dim": 8, "hidden_dims": [16, 8], "rho_min": 0.25},
        "filter": {
            "initial_attitude_error_deg": [0.1, 0.1, 0.5],
            "initial_bias_error_dph": [0.01, 0.01, 0.01],
            "process_attitude_std_deg": 0.015,
            "process_bias_std_dph": 0.002,
            "gps_attitude_std_deg": 0.35,
            "vi_velocity_std": 0.08,
            "vi_attitude_std_deg": 0.20,
            "convergence_threshold_deg": 0.8,
            "convergence_hold_s": 2.0,
            "failure_heading_threshold_deg": 3.0,
        },
        "experiment": {"scenario": "degraded_gps", "evaluation_window_s": [0.0, 8.0], "methods": ["gps_sins_closed_loop", "vi_empirical"]},
    }
    data_root = generate_toy_dataset(cfg, tmp_path / "data", seed=1)
    df = evaluate_dataset(data_root, cfg, tmp_path / "out", checkpoint=None, seed=1)
    assert set(df["method"]) == {"gps_sins_closed_loop", "vi_empirical"}

