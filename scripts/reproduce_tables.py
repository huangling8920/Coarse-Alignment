from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.runner import reproduce_tables
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    df = reproduce_tables(cfg, args.output)
    print(df.groupby("method")["heading_rmse_deg"].agg(["mean", "std"]).to_string())


if __name__ == "__main__":
    main()

