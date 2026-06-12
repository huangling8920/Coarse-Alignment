from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.utils.io import read_csv


@dataclass
class SensorSequence:
    root: Path
    truth: pd.DataFrame
    imu: pd.DataFrame
    gps: pd.DataFrame
    vi: pd.DataFrame
    quality_train: pd.DataFrame
    calibration: dict

    @classmethod
    def load(cls, root: str | Path) -> "SensorSequence":
        root = Path(root)
        calib_path = root / "calibration.yaml"
        if not calib_path.exists():
            raise FileNotFoundError(f"Missing calibration file: {calib_path}")
        with calib_path.open("r", encoding="utf-8") as f:
            calibration = yaml.safe_load(f) or {}
        return cls(
            root=root,
            truth=read_csv(root / "truth.csv"),
            imu=read_csv(root / "imu.csv"),
            gps=read_csv(root / "gps.csv"),
            vi=read_csv(root / "vi_measurements.csv"),
            quality_train=read_csv(root / "quality_train.csv"),
            calibration=calibration,
        )

    def truth_heading_at(self, times: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(times, dtype=float), self.truth["t"].to_numpy(), self.truth["heading_deg"].to_numpy())

    def truth_bias_at(self, times: np.ndarray) -> np.ndarray:
        cols = ["bgx_dph", "bgy_dph", "bgz_dph"]
        return np.vstack(
            [np.interp(np.asarray(times, dtype=float), self.truth["t"].to_numpy(), self.truth[c].to_numpy()) for c in cols]
        ).T

