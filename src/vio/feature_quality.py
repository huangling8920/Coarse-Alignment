from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QualityFeatures:
    n_feat_ratio: float
    track_success: float
    inlier_ratio: float
    feature_dispersion: float
    reproj_norm: float
    blur_indicator: float
    reflection_indicator: float
    preint_norm: float

    def as_array(self) -> np.ndarray:
        arr = np.array(
            [
                self.n_feat_ratio,
                self.track_success,
                self.inlier_ratio,
                self.feature_dispersion,
                self.reproj_norm,
                self.blur_indicator,
                self.reflection_indicator,
                self.preint_norm,
            ],
            dtype=float,
        )
        if arr.shape != (8,):
            raise ValueError("quality feature vector must have dimension 8")
        return arr


def quality_features_from_row(row, n_ref_features: float, reproj_ref_px: float = 2.0, preint_ref: float = 0.08) -> QualityFeatures:
    return QualityFeatures(
        n_feat_ratio=float(row["feature_count"]) / float(n_ref_features),
        track_success=float(row["track_success"]),
        inlier_ratio=float(row["inlier_ratio"]),
        feature_dispersion=float(row["feature_dispersion"]),
        reproj_norm=float(row["reproj_error_px"]) / float(reproj_ref_px),
        blur_indicator=float(row["blur_indicator"]),
        reflection_indicator=float(row["reflection_indicator"]),
        preint_norm=float(row["preint_residual"]) / float(preint_ref),
    )


def empirical_quality(features: np.ndarray, rho_min: float = 0.25) -> float:
    """Rule-based quality used for ablation.

    Inputs follow the 8-D manuscript feature vector. Larger reprojection,
    blur, reflection, and preintegration residual reduce the quality.
    """
    f = np.asarray(features, dtype=float).reshape(8)
    good = 0.28 * f[0] + 0.22 * f[1] + 0.22 * f[2] + 0.10 * f[3]
    bad = 0.08 * f[4] + 0.05 * f[5] + 0.03 * f[6] + 0.02 * f[7]
    rho = good - bad
    return float(np.clip(rho_min + (1.0 - rho_min) * rho, rho_min, 1.0))


def degradation_profile(level: str) -> dict[str, float]:
    profiles = {
        "normal": dict(feature_count=186, track_success=0.94, inlier_ratio=0.91, feature_dispersion=0.86, reproj_error_px=0.62, blur_indicator=0.05, reflection_indicator=0.05, preint_residual=0.018, rho_target=0.90),
        "weak_snow": dict(feature_count=92, track_success=0.66, inlier_ratio=0.61, feature_dispersion=0.55, reproj_error_px=1.18, blur_indicator=0.18, reflection_indicator=0.12, preint_residual=0.038, rho_target=0.56),
        "strong_reflection": dict(feature_count=74, track_success=0.58, inlier_ratio=0.53, feature_dispersion=0.48, reproj_error_px=1.46, blur_indicator=0.12, reflection_indicator=0.78, preint_residual=0.046, rho_target=0.43),
        "motion_blur": dict(feature_count=61, track_success=0.49, inlier_ratio=0.45, feature_dispersion=0.42, reproj_error_px=1.72, blur_indicator=0.82, reflection_indicator=0.18, preint_residual=0.057, rho_target=0.35),
    }
    if level not in profiles:
        raise KeyError(f"Unknown visual degradation level: {level}")
    return profiles[level].copy()

