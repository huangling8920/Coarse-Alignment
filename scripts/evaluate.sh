#!/usr/bin/env bash
set -euo pipefail
python scripts/evaluate.py --config configs/default.yaml --data data/toy --checkpoint outputs/default/checkpoints/quality_mlp.pt --output outputs/default_eval

