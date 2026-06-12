#!/usr/bin/env bash
set -euo pipefail
python scripts/run_ablation.py --config configs/ablation.yaml --data data/toy --checkpoint outputs/default/checkpoints/quality_mlp.pt --output outputs/ablation

