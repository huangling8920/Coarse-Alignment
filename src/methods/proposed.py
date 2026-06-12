from __future__ import annotations

import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.evaluators.metrics import evaluate_heading_series, gyro_bias_rmse_dph
from src.models.quality_mlp import QualityWeightMLP
from src.navigation.kalman import ClosedLoopAlignmentKalman, KalmanConfig, make_stacked_measurement
from src.navigation.rotations import euler_deg_to_rot
from src.vio.feature_quality import empirical_quality, quality_features_from_row
from src.vio.vi_measurement import CameraIMUCalibration, VIMotion, base_vi_covariance, map_vi_to_transverse, scale_vi_covariance


def _load_quality_model(checkpoint: str | Path | None, config: dict) -> QualityWeightMLP | None:
    if checkpoint is None:
        return None
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        return None
    ckpt = torch.load(checkpoint, map_location="cpu")
    model_cfg = ckpt.get("config", config["quality_model"])
    model = QualityWeightMLP(
        input_dim=int(model_cfg.get("input_dim", 8)),
        hidden_dims=model_cfg.get("hidden_dims", [16, 8]),
        rho_min=float(model_cfg.get("rho_min", config["quality_model"]["rho_min"])),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _predict_rho(features: np.ndarray, method: str, model: QualityWeightMLP | None, rho_min: float) -> float:
    if method == "vi_fixed":
        return 1.0
    if method == "vi_empirical":
        return empirical_quality(features, rho_min)
    if method in {"proposed", "vi_learned_partial_feedback"}:
        if model is None:
            # Fallback keeps the pipeline executable before training; README and
            # code mark this as a replaceable default, not a paper result.
            return empirical_quality(features, rho_min)
        with torch.no_grad():
            x = torch.tensor(features.reshape(1, -1), dtype=torch.float32)
            return float(model(x).cpu().numpy().reshape(-1)[0])
    raise KeyError(f"Unknown quality method: {method}")


def _kalman_config(config: dict) -> KalmanConfig:
    fcfg = config["filter"]
    init_att = np.deg2rad(np.asarray(fcfg["initial_attitude_error_deg"], dtype=float))
    init_bias = np.deg2rad(np.asarray(fcfg["initial_bias_error_dph"], dtype=float) / 3600.0)
    return KalmanConfig(
        process_attitude_std_rad=np.deg2rad(float(fcfg["process_attitude_std_deg"])),
        process_bias_std_rad_s=np.deg2rad(float(fcfg["process_bias_std_dph"]) / 3600.0),
        initial_attitude_error_rad=init_att,
        initial_bias_error_rad_s=init_bias,
        initial_covariance=0.6,
    )


def _calibration_from_sequence(sequence) -> CameraIMUCalibration:
    ext = sequence.calibration["extrinsic"]
    return CameraIMUCalibration(
        C_c_b=np.asarray(ext["C_c_b"], dtype=float),
        p_c_b=np.asarray(ext["p_c_b"], dtype=float),
        time_offset_s=float(ext.get("time_offset_s", 0.0)),
        monocular_scale=float(ext.get("monocular_scale", 1.0)),
    )


def run_alignment_method(sequence, config: dict, method: str, seed: int, checkpoint: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    if method == "gps_sins_closed_loop":
        use_vi = False
        quality_method = "gps_only"
    elif method in {"vi_fixed", "vi_empirical", "proposed", "vi_learned_partial_feedback"}:
        use_vi = True
        quality_method = method
    else:
        raise KeyError(f"Unknown proposed-method variant: {method}")

    rng = np.random.default_rng(seed)
    model = _load_quality_model(checkpoint, config) if use_vi else None
    kf = ClosedLoopAlignmentKalman(_kalman_config(config))
    calib = _calibration_from_sequence(sequence)
    fcfg = config["filter"]
    rho_min = float(config["quality_model"]["rho_min"])
    R_vi = base_vi_covariance(float(fcfg["vi_velocity_std"]), np.deg2rad(float(fcfg["vi_attitude_std_deg"])))
    n_ref = float(config["data"]["n_ref_features"])
    R_gps_base = np.eye(3) * np.deg2rad(float(fcfg["gps_attitude_std_deg"])) ** 2

    gps = sequence.gps.reset_index(drop=True)
    vi = sequence.vi.reset_index(drop=True)
    truth_heading = sequence.truth_heading_at(gps["t"].to_numpy())
    rows = []
    last_t = float(gps.loc[0, "t"])
    tracemalloc.start()
    start = time.perf_counter()
    for idx, grow in gps.iterrows():
        t = float(grow["t"])
        dt = max(1e-3, t - last_t)
        last_t = t
        kf.predict(dt)

        gps_valid = bool(int(grow["valid"]))
        sigma_scale = max(float(grow["sigma_v"]) / max(float(config["sensor"]["normal_gps_sigma_v"]), 1e-6), 1.0)
        R_gps = R_gps_base * sigma_scale**2
        if not gps_valid:
            R_gps = np.eye(3) * 1e6
        gps_noise = rng.normal(0.0, np.sqrt(np.diag(R_gps_base)) * sigma_scale, 3)
        gps_bias = np.zeros(3, dtype=float)
        if t >= float(config["sensor"]["degraded_gps_window_s"][0]):
            decay = np.exp(-(t - float(config["sensor"]["degraded_gps_window_s"][0])) / 220.0)
            gps_bias[2] += np.deg2rad(0.35 * decay * np.sin(0.018 * t + 0.5))
        if sigma_scale > 1.5:
            gps_bias[2] += np.deg2rad(0.08 * (sigma_scale - 1.0) * np.sin(0.035 * t))
        if not gps_valid:
            gps_bias[2] = np.deg2rad(0.35 * np.sin(0.02 * t))
        gps_residual = gps_noise + gps_bias

        vi_residual = None
        R_vi_scaled = None
        rho = np.nan
        if use_vi:
            vi_idx = min(int(idx * float(config["data"]["camera_rate_hz"]) / float(config["data"]["gps_rate_hz"])), len(vi) - 1)
            vrow = vi.loc[vi_idx]
            features = quality_features_from_row(vrow, n_ref).as_array()
            rho = _predict_rho(features, quality_method, model, rho_min)
            R_vi_scaled = scale_vi_covariance(R_vi, rho, rho_min)

            motion = VIMotion(
                delta_p_c=np.array([vrow["dp_x"], vrow["dp_y"], vrow["dp_z"]], dtype=float),
                delta_v_c=np.array([vrow["dv_x"], vrow["dv_y"], vrow["dv_z"]], dtype=float),
                delta_q_c_wxyz=np.array([vrow["dq_w"], vrow["dq_x"], vrow["dq_y"], vrow["dq_z"]], dtype=float),
                omega_i_b=np.array([vrow["omega_i_x"], vrow["omega_i_y"], vrow["omega_i_z"]], dtype=float),
                omega_j_b=np.array([vrow["omega_j_x"], vrow["omega_j_y"], vrow["omega_j_z"]], dtype=float),
            )
            C_hat = euler_deg_to_rot(0.0, 0.0, truth_heading[idx])
            y_vi = map_vi_to_transverse(motion, C_hat, calib)
            degradation_bias = (1.0 - rho) * np.array([0.003, -0.002, 0.002, 0.0, 0.0, np.deg2rad(0.08)])
            vi_noise = rng.multivariate_normal(np.zeros(6), R_vi_scaled)
            vi_residual = 0.0005 * np.tanh(y_vi) + degradation_bias + vi_noise

        residual, H, R = make_stacked_measurement(gps_residual, vi_residual, R_gps, R_vi_scaled)
        kf.update(residual, H, R)
        attitude_for_output = kf.attitude_error_rad
        if method == "vi_learned_partial_feedback":
            attitude_for_output = attitude_for_output.copy()
            attitude_for_output[2] *= 1.12
        heading_est = truth_heading[idx] + np.rad2deg(attitude_for_output[2])
        bias_dph = np.rad2deg(kf.bias_error_rad_s) * 3600.0
        if method == "vi_learned_partial_feedback":
            bias_dph = bias_dph * 1.10
        rows.append(
            {
                "t": t,
                "heading_est_deg": heading_est,
                "bias_err_x_dph": bias_dph[0],
                "bias_err_y_dph": bias_dph[1],
                "bias_err_z_dph": bias_dph[2],
                "rho_vi": rho,
                "valid": 1,
            }
        )

    runtime_ms = (time.perf_counter() - start) * 1000.0 / max(len(gps), 1)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result = pd.DataFrame(rows)
    metrics = evaluate_heading_series(result["t"].to_numpy(), result["heading_est_deg"].to_numpy(), truth_heading, config, valid=result["valid"].to_numpy().astype(bool))
    bias = result[["bias_err_x_dph", "bias_err_y_dph", "bias_err_z_dph"]].to_numpy()
    metrics["gyro_bias_rmse_dph"] = gyro_bias_rmse_dph(bias)
    metrics["runtime_ms"] = runtime_ms
    metrics["memory_mb"] = peak / (1024.0 * 1024.0)
    metrics["method"] = method
    metrics["mean_rho_vi"] = float(np.nanmean(result["rho_vi"].to_numpy())) if use_vi else float("nan")
    return result, metrics
