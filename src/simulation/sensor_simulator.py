from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from src.simulation.polar_trajectory import generate_reference_truth, generate_velocity
from src.utils.io import ensure_dir, write_csv
from src.vio.feature_quality import degradation_profile


def _is_outage(t: float, outages: list[list[float]]) -> bool:
    return any(float(a) <= t <= float(b) for a, b in outages)


def _visual_level_for_time(t: float, scenario: str) -> str:
    if scenario == "visual_degradation":
        if t < 75:
            return "normal"
        if t < 150:
            return "weak_snow"
        if t < 225:
            return "strong_reflection"
        return "motion_blur"
    if scenario == "vr_s3":
        return "motion_blur" if 120 <= t <= 190 else "weak_snow"
    if scenario == "vr_s2":
        return "strong_reflection"
    if scenario == "vr_s4":
        return "weak_snow"
    return "normal"


def _draw_toy_images(output_dir: Path, n_frames: int, width: int, height: int, scenario: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    img_dir = ensure_dir(output_dir / "images")
    for k in range(n_frames):
        level = _visual_level_for_time(k / 20.0, scenario)
        base = np.full((height, width, 3), 235, dtype=np.uint8)
        if level == "strong_reflection":
            base[:, :] = np.array([215, 230, 245], dtype=np.uint8)
        elif level == "motion_blur":
            base[:, :] = np.array([225, 225, 225], dtype=np.uint8)
        img = Image.fromarray(base)
        draw = ImageDraw.Draw(img)
        n_marks = {"normal": 80, "weak_snow": 35, "strong_reflection": 45, "motion_blur": 25}[level]
        for _ in range(n_marks):
            x = int(rng.integers(0, width))
            y = int(rng.integers(0, height))
            shade = int(rng.integers(80, 180))
            draw.ellipse((x, y, x + 2, y + 2), fill=(shade, shade, shade))
        if level == "strong_reflection":
            for _ in range(8):
                x = int(rng.integers(0, width - 80))
                y = int(rng.integers(0, height - 20))
                draw.rectangle((x, y, x + 80, y + 8), fill=(255, 255, 255))
        img.save(img_dir / f"frame_{k:06d}.png")


def generate_toy_dataset(config: dict, output: str | Path, scenario: str | None = None, seed: int | None = None) -> Path:
    output = ensure_dir(output)
    data_cfg = config["data"]
    sensor_cfg = config["sensor"]
    cal_cfg = config["calibration"]
    scenario = scenario or config.get("experiment", {}).get("scenario", "degraded_gps")
    seed = int(seed if seed is not None else config["project"]["seed"])
    rng = np.random.default_rng(seed)

    duration = float(data_cfg["duration_s"])
    gps_rate = float(data_cfg["gps_rate_hz"])
    imu_rate = float(data_cfg["imu_rate_hz"])
    cam_rate = float(data_cfg["camera_rate_hz"])
    width = int(data_cfg["image_width"])
    height = int(data_cfg["image_height"])

    truth = generate_reference_truth(duration, gps_rate, seed)
    write_csv(truth, output / "truth.csv")

    t_imu = np.arange(0.0, duration + 1e-9, 1.0 / imu_rate)
    omega = np.vstack(
        [
            0.002 * np.sin(0.03 * t_imu),
            0.002 * np.cos(0.027 * t_imu),
            0.015 + 0.001 * np.sin(0.012 * t_imu),
        ]
    ).T
    gyro_bias_rad_s = np.deg2rad(float(sensor_cfg["gyro_bias_dph"]) / 3600.0)
    omega += gyro_bias_rad_s + rng.normal(0.0, 2e-5, omega.shape)
    accel = np.vstack(
        [
            0.02 * np.sin(0.04 * t_imu),
            0.03 * np.cos(0.031 * t_imu),
            -9.81 + 0.02 * np.sin(0.01 * t_imu),
        ]
    ).T
    accel += rng.normal(0.0, 0.01, accel.shape)
    imu = pd.DataFrame({"t": t_imu, "wx": omega[:, 0], "wy": omega[:, 1], "wz": omega[:, 2], "ax": accel[:, 0], "ay": accel[:, 1], "az": accel[:, 2]})
    write_csv(imu, output / "imu.csv")

    t_gps = truth["t"].to_numpy()
    vel = generate_velocity(t_gps)
    gps_rows = []
    outages = sensor_cfg.get("gps_outages_s", [])
    for idx, t in enumerate(t_gps):
        degraded = float(sensor_cfg["degraded_gps_window_s"][0]) <= t <= float(sensor_cfg["degraded_gps_window_s"][1])
        sigma = float(sensor_cfg["degraded_gps_sigma_v"] if degraded else sensor_cfg["normal_gps_sigma_v"])
        if scenario == "vr_s2":
            sigma = 0.20
        valid = not _is_outage(float(t), outages)
        noisy = vel[idx] + rng.normal(0.0, sigma, 3)
        gps_rows.append([t, *noisy, int(valid), sigma])
    gps = pd.DataFrame(gps_rows, columns=["t", "vx", "vy", "vz", "valid", "sigma_v"])
    write_csv(gps, output / "gps.csv")

    t_cam = np.arange(0.0, duration + 1e-9, 1.0 / cam_rate)
    vi_rows = []
    q_train_rows = []
    n_ref = float(data_cfg["n_ref_features"])
    for i in range(len(t_cam) - 1):
        ti, tj = float(t_cam[i]), float(t_cam[i + 1])
        level = _visual_level_for_time(ti, scenario)
        prof = degradation_profile(level)
        feature_count = max(5, int(rng.normal(prof["feature_count"], 6)))
        track = float(np.clip(rng.normal(prof["track_success"], 0.025), 0.05, 1.0))
        inlier = float(np.clip(rng.normal(prof["inlier_ratio"], 0.025), 0.05, 1.0))
        disp = float(np.clip(rng.normal(prof["feature_dispersion"], 0.025), 0.05, 1.0))
        reproj = float(max(0.05, rng.normal(prof["reproj_error_px"], 0.06)))
        blur = float(np.clip(rng.normal(prof["blur_indicator"], 0.03), 0.0, 1.0))
        reflection = float(np.clip(rng.normal(prof["reflection_indicator"], 0.03), 0.0, 1.0))
        preint = float(max(0.001, rng.normal(prof["preint_residual"], 0.003)))
        quality_noise = 1.0 + (1.0 - prof["rho_target"]) * 2.0
        dv = generate_velocity(np.array([tj]))[0] * (tj - ti) + rng.normal(0.0, 0.006 * quality_noise, 3)
        dp = dv * (tj - ti)
        yaw_delta = np.deg2rad(0.05 * np.sin(0.01 * ti)) + rng.normal(0.0, np.deg2rad(0.01 * quality_noise))
        dq = Rotation.from_rotvec([0.0, 0.0, yaw_delta]).as_quat()
        q_wxyz = [dq[3], dq[0], dq[1], dq[2]]
        omega_i = omega[min(int(ti * imu_rate), len(omega) - 1)]
        omega_j = omega[min(int(tj * imu_rate), len(omega) - 1)]
        vi_rows.append(
            [
                ti,
                tj,
                *dp,
                *dv,
                *q_wxyz,
                *omega_i,
                *omega_j,
                feature_count,
                track,
                inlier,
                disp,
                reproj,
                blur,
                reflection,
                preint,
            ]
        )
        rho_target = float(np.clip(rng.normal(prof["rho_target"], 0.025), 0.25, 1.0))
        q_train_rows.append([feature_count / n_ref, track, inlier, disp, reproj / 2.0, blur, reflection, preint / 0.08, rho_target])

    vi_cols = [
        "t_i",
        "t_j",
        "dp_x",
        "dp_y",
        "dp_z",
        "dv_x",
        "dv_y",
        "dv_z",
        "dq_w",
        "dq_x",
        "dq_y",
        "dq_z",
        "omega_i_x",
        "omega_i_y",
        "omega_i_z",
        "omega_j_x",
        "omega_j_y",
        "omega_j_z",
        "feature_count",
        "track_success",
        "inlier_ratio",
        "feature_dispersion",
        "reproj_error_px",
        "blur_indicator",
        "reflection_indicator",
        "preint_residual",
    ]
    quality_cols = [
        "n_feat_ratio",
        "track_success",
        "inlier_ratio",
        "feature_dispersion",
        "reproj_norm",
        "blur_indicator",
        "reflection_indicator",
        "preint_norm",
        "rho_target",
    ]
    write_csv(pd.DataFrame(vi_rows, columns=vi_cols), output / "vi_measurements.csv")
    write_csv(pd.DataFrame(q_train_rows, columns=quality_cols), output / "quality_train.csv")

    calibration = {
        "camera": {
            "width": width,
            "height": height,
            "fx": 376.0,
            "fy": 376.0,
            "cx": 376.0,
            "cy": 240.0,
            "distortion": [0, 0, 0, 0],
        },
        "extrinsic": {
            "C_c_b": cal_cfg["C_c_b"],
            "p_c_b": cal_cfg["p_c_b"],
            "time_offset_s": cal_cfg["time_offset_s"],
            "monocular_scale": cal_cfg["monocular_scale"],
        },
    }
    with (output / "calibration.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(calibration, f, sort_keys=False)

    _draw_toy_images(output, min(80, int(duration * cam_rate)), width, height, scenario, seed)
    return output

