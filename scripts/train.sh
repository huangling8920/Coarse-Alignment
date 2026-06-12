#!/usr/bin/env bash
set -euo pipefail
python scripts/generate_toy_data.py --config configs/default.yaml --output data/toy
python scripts/train_quality.py --config configs/default.yaml --data data/toy --output outputs/default

