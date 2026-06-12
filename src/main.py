from __future__ import annotations

import argparse
from pathlib import Path

from src.experiments.runner import evaluate_dataset, reproduce_tables
from src.simulation.sensor_simulator import generate_toy_dataset
from src.trainers.quality_trainer import train_quality_model
from src.utils.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quality-aware VI GPS/SINS reproduction entry point")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate-data")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--scenario", default=None)
    p.add_argument("--seed", type=int, default=None)

    p = sub.add_parser("train-quality")
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)

    p = sub.add_parser("evaluate")
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=None)

    p = sub.add_parser("reproduce-tables")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    if args.command == "generate-data":
        path = generate_toy_dataset(cfg, args.output, args.scenario, args.seed)
        print(f"Generated toy dataset: {Path(path).resolve()}")
    elif args.command == "train-quality":
        info = train_quality_model(args.data, cfg, args.output)
        print(info)
    elif args.command == "evaluate":
        df = evaluate_dataset(args.data, cfg, args.output, args.checkpoint, args.seed)
        print(df.to_string(index=False))
    elif args.command == "reproduce-tables":
        df = reproduce_tables(cfg, args.output)
        print(df.groupby("method")["heading_rmse_deg"].mean().to_string())


if __name__ == "__main__":
    main()

