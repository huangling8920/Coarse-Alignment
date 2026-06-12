from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluators.metrics import evaluate_heading_series


REQUIRED_COLUMNS = {"t", "roll_deg", "pitch_deg", "heading_deg", "valid"}


def load_external_vio_result(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"External VIO result is missing columns: {sorted(missing)}")
    return df


def evaluate_external_vio(path: str | Path, truth: pd.DataFrame, config: dict) -> dict:
    df = load_external_vio_result(path)
    merged = pd.merge_asof(
        df.sort_values("t"),
        truth[["t", "heading_deg"]].sort_values("t"),
        on="t",
        suffixes=("_est", "_truth"),
        direction="nearest",
    )
    metrics = evaluate_heading_series(
        merged["t"].to_numpy(),
        merged["heading_deg_est"].to_numpy(),
        merged["heading_deg_truth"].to_numpy(),
        config,
        valid=merged["valid"].to_numpy().astype(bool),
    )
    if "runtime_ms" in df:
        metrics["runtime_ms"] = float(df["runtime_ms"].mean())
    if "memory_mb" in df:
        metrics["memory_mb"] = float(df["memory_mb"].max())
    return metrics

