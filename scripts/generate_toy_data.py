from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.sensor_simulator import generate_toy_dataset
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    path = generate_toy_dataset(cfg, args.output, args.scenario, args.seed)
    print(f"Generated toy dataset at {Path(path).resolve()}")


if __name__ == "__main__":
    main()

