from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, help="Path to metrics_summary.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.metrics)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["method"], df["heading_rmse_deg"])
    ax.set_ylabel("Heading RMSE (deg)")
    ax.set_xlabel("Method")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out / "heading_rmse_bar.png", dpi=200)
    print(f"Saved {out / 'heading_rmse_bar.png'}")


if __name__ == "__main__":
    main()

