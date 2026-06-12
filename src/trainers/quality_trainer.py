from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.losses.quality_loss import innovation_nll_proxy_loss
from src.models.quality_mlp import QualityWeightMLP, count_parameters
from src.utils.config import save_config
from src.utils.io import ensure_dir
from src.utils.seeding import seed_everything


FEATURE_COLUMNS = [
    "n_feat_ratio",
    "track_success",
    "inlier_ratio",
    "feature_dispersion",
    "reproj_norm",
    "blur_indicator",
    "reflection_indicator",
    "preint_norm",
]


def _device_from_config(config: dict) -> torch.device:
    requested = config.get("runtime", {}).get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train_quality_model(data_root: str | Path, config: dict, output_dir: str | Path) -> dict:
    seed_everything(int(config["project"]["seed"]))
    output_dir = ensure_dir(output_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    df = pd.read_csv(Path(data_root) / "quality_train.csv")
    missing = set(FEATURE_COLUMNS + ["rho_target"]) - set(df.columns)
    if missing:
        raise ValueError(f"quality_train.csv missing columns: {sorted(missing)}")

    x = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df[["rho_target"]].to_numpy(dtype=np.float32)
    heading_proxy = ((1.0 - y) ** 2).astype(np.float32)
    n = len(df)
    indices = np.arange(n)
    rng = np.random.default_rng(int(config["project"]["seed"]))
    rng.shuffle(indices)
    val_ratio = float(config["quality_model"]["val_ratio"])
    test_ratio = float(config["quality_model"]["test_ratio"])
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]

    device = _device_from_config(config)
    model = QualityWeightMLP(
        input_dim=int(config["quality_model"]["input_dim"]),
        hidden_dims=config["quality_model"]["hidden_dims"],
        rho_min=float(config["quality_model"]["rho_min"]),
    ).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(config["quality_model"]["lr"]),
        weight_decay=float(config["quality_model"].get("weight_decay", 0.0)),
    )
    train_ds = TensorDataset(
        torch.from_numpy(x[train_idx]),
        torch.from_numpy(y[train_idx]),
        torch.from_numpy(heading_proxy[train_idx]),
    )
    loader = DataLoader(train_ds, batch_size=int(config["quality_model"]["batch_size"]), shuffle=True)
    start = time.perf_counter()
    history = []
    for epoch in range(int(config["quality_model"]["epochs"])):
        model.train()
        losses = []
        for xb, yb, hb in loader:
            xb, yb, hb = xb.to(device), yb.to(device), hb.to(device)
            pred = model(xb)
            loss = innovation_nll_proxy_loss(
                pred,
                yb,
                hb,
                heading_proxy_weight=float(config["quality_model"]["heading_proxy_weight"]),
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses))})

    def _mae(idx: np.ndarray) -> float:
        if idx.size == 0:
            return float("nan")
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(x[idx]).to(device)).cpu().numpy()
        return float(np.mean(np.abs(pred - y[idx])))

    ckpt_path = ckpt_dir / "quality_mlp.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config["quality_model"],
            "feature_columns": FEATURE_COLUMNS,
        },
        ckpt_path,
    )
    save_config(config, output_dir / "config_resolved.yaml")
    pd.DataFrame(history).to_csv(output_dir / "quality_training_log.csv", index=False)
    if Path(data_root, "calibration.yaml").exists():
        shutil.copy2(Path(data_root, "calibration.yaml"), output_dir / "calibration.yaml")

    return {
        "checkpoint": str(ckpt_path),
        "num_parameters": count_parameters(model),
        "train_samples": int(train_idx.size),
        "val_mae": _mae(val_idx),
        "test_mae": _mae(test_idx),
        "train_time_s": time.perf_counter() - start,
        "device": str(device),
    }

