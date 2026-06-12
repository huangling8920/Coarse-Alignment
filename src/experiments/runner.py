from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.datasets.sensor_sequence import SensorSequence
from src.evaluators.tables import save_metric_tables
from src.methods.baselines import evaluate_baseline_result, run_protocol_baseline
from src.methods.proposed import run_alignment_method
from src.simulation.sensor_simulator import generate_toy_dataset
from src.trainers.quality_trainer import train_quality_model
from src.utils.config import save_config
from src.utils.io import ensure_dir
from src.utils.seeding import seed_everything


PROTOCOL_BASELINES = {"transverse_oba", "transverse_usque"}


def evaluate_dataset(
    data_root: str | Path,
    config: dict,
    output_dir: str | Path,
    checkpoint: str | Path | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    seed = int(seed if seed is not None else config["project"]["seed"])
    seed_everything(seed)
    output_dir = ensure_dir(output_dir)
    sequence = SensorSequence.load(data_root)
    methods = list(config.get("experiment", {}).get("methods", ["proposed"]))
    metric_rows = []
    for method in methods:
        method_dir = ensure_dir(output_dir / method)
        if method in PROTOCOL_BASELINES:
            result = run_protocol_baseline(sequence, config, method, seed)
            metrics = evaluate_baseline_result(result, sequence, config, method)
        else:
            result, metrics = run_alignment_method(sequence, config, method, seed, checkpoint)
        result.to_csv(method_dir / f"result_seed_{seed}.csv", index=False)
        metrics["seed"] = seed
        metric_rows.append(metrics)
    metrics_df = pd.DataFrame(metric_rows)
    save_metric_tables(metrics_df, output_dir)
    save_config(config, output_dir / "config_resolved.yaml")
    return metrics_df


def reproduce_tables(config: dict, output_dir: str | Path) -> pd.DataFrame:
    output_dir = ensure_dir(output_dir)
    datasets_dir = ensure_dir(output_dir / "datasets")
    train_data = generate_toy_dataset(config, datasets_dir / "train", config.get("experiment", {}).get("scenario", "degraded_gps"), int(config["project"]["seed"]))
    train_info = train_quality_model(train_data, config, output_dir)
    checkpoint = train_info["checkpoint"]

    rows = []
    for seed in config["data"]["seeds"]:
        data_root = generate_toy_dataset(config, datasets_dir / f"seed_{seed}", config.get("experiment", {}).get("scenario", "degraded_gps"), int(seed))
        df = evaluate_dataset(data_root, config, output_dir / f"seed_{seed}", checkpoint, int(seed))
        rows.append(df)
    all_metrics = pd.concat(rows, ignore_index=True)
    tables_dir = ensure_dir(output_dir / "tables")
    save_metric_tables(all_metrics, tables_dir)
    pd.DataFrame([train_info]).to_csv(output_dir / "quality_model_summary.csv", index=False)
    return all_metrics

