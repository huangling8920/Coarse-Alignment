from __future__ import annotations

from pathlib import Path

import pandas as pd


def summarize_metrics(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in per_seed.groupby("method"):
        rows.append(
            {
                "method": method,
                "mean_error_deg": group["mean_error_deg"].mean(),
                "std_error_deg": group["std_error_deg"].mean(),
                "heading_rmse_deg": group["heading_rmse_deg"].mean(),
                "heading_rmse_std_deg": group["heading_rmse_deg"].std(ddof=0),
                "max_heading_error_deg": group["max_heading_error_deg"].mean(),
                "convergence_time_s": group["convergence_time_s"].mean(),
                "failure_rate": group["failure"].mean(),
                "gyro_bias_rmse_dph": group.get("gyro_bias_rmse_dph", pd.Series(dtype=float)).mean(),
                "runtime_ms": group.get("runtime_ms", pd.Series(dtype=float)).mean(),
                "memory_mb": group.get("memory_mb", pd.Series(dtype=float)).mean(),
            }
        )
    return pd.DataFrame(rows)


def save_metric_tables(per_seed: pd.DataFrame, output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_path = output_dir / "metrics_per_seed.csv"
    summary_path = output_dir / "metrics_summary.csv"
    per_seed.to_csv(per_seed_path, index=False)
    summarize_metrics(per_seed).to_csv(summary_path, index=False)
    return per_seed_path, summary_path

