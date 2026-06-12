**Quality-Aware Visual-Inertial GPS/SINS Coarse Alignment in a Transverse Frame for Robust Polar Navigation**


## Method-to-Code Map

| Paper component | Code |
| --- | --- |
| Transverse-frame attitude and velocity-vector tools | `src/navigation/frames.py`, `src/navigation/rotations.py`, `src/navigation/velocity_vectors.py` |
| 6-state closed-loop Kalman update with 9-D stacked measurement | `src/navigation/kalman.py` |
| Visual-inertial relative motion to transverse-frame auxiliary observation | `src/vio/vi_measurement.py` |
| Feature statistics and visual quality descriptors | `src/vio/feature_quality.py` |
| Learned visual-inertial quality weight rho_VI | `src/models/quality_mlp.py`, `src/losses/quality_loss.py` |
| Proposed alignment method and ablations | `src/methods/proposed.py`, `src/methods/baselines.py` |
| Synthetic polar sensor sequences | `src/simulation/` |
| Evaluation metrics and result tables | `src/evaluators/metrics.py`, `src/evaluators/tables.py` |
| Training/evaluation entry points | `src/main.py`, `scripts/*.py` |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data Preparation

```bash
python scripts/generate_toy_data.py --config configs/default.yaml --output data/toy
```

The generated toy package contains:

- `truth.csv`
- `imu.csv`
- `gps.csv`
- `vi_measurements.csv`
- `quality_train.csv`
- `calibration.yaml`
- toy PNG image sequences under `images/`

Real Unity or vehicle-test data can replace the toy data if it follows the schema described in `data/README.md`.

## Train the Learned Quality Model

```bash
python scripts/train_quality.py --config configs/default.yaml --data data/toy --output outputs/default
```

The trained model and copied config are saved under `outputs/default/checkpoints/`.

## Evaluate

```bash
python scripts/evaluate.py --config configs/default.yaml --data data/toy --checkpoint outputs/default/checkpoints/quality_mlp.pt --output outputs/default
```

## Ablation

```bash
python scripts/run_ablation.py --config configs/ablation.yaml --data data/toy --checkpoint outputs/default/checkpoints/quality_mlp.pt --output outputs/ablation
```

## Reproduce Tables

```bash
python scripts/reproduce_tables.py --config configs/default.yaml --output outputs/reproduction
```

This command regenerates toy data, trains the quality model, runs the selected methods across seeds/scenarios, and writes CSV tables to `outputs/reproduction/tables/`.

## External VIO Baselines

This repository does not reimplement VINS-Mono, OpenVINS, or ORB-SLAM3. Instead, it provides a standard CSV adapter in `src/vio/external_adapters.py`. To compare against these baselines, export their results with columns:

```text
t, roll_deg, pitch_deg, heading_deg, valid, runtime_ms, memory_mb
```

The evaluator converts them into the same heading-error, convergence, failure, runtime, and memory metrics.

## Outputs

Each run saves:

- resolved configuration copy;
- per-seed metrics CSV;
- aggregate metrics CSV;
- model checkpoint if training is used;
- runtime and memory summary;
- optional plots generated from logs.

## Tests

```bash
pytest -q
```

The tests check tensor dimensions, quality-model forward pass, Kalman update shapes, metric definitions, and the toy pipeline smoke path.

