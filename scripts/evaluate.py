"""Evaluate a trained MTFF-Net checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from mtff_net.config import load_config
from mtff_net.data import chronological_split, load_dataset
from mtff_net.metrics import grouped_metrics, per_target_metrics
from mtff_net.models import MTTFNet
from mtff_net.train import predict, transform_X


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("outputs/dataset/dataset_arrays.npz"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = load_dataset(args.dataset)
    train_sl, val_sl, test_sl = chronological_split(len(dataset.X), cfg.dataset.train_ratio, cfg.dataset.val_ratio)
    X_train_raw = dataset.X[train_sl]
    y_train_raw = dataset.y[train_sl]
    X_test_raw = dataset.X[test_sl]
    y_test_raw = dataset.y[test_sl]

    x_scaler = StandardScaler().fit(X_train_raw.reshape(-1, X_train_raw.shape[-1]))
    y_scaler = StandardScaler().fit(y_train_raw)
    X_test = transform_X(X_test_raw, x_scaler)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    n_crack = sum(col.startswith("crack_") for col in dataset.target_columns)
    n_rebar = sum(col.startswith("rebar_") for col in dataset.target_columns)
    model_config = checkpoint.get("model_config", vars(cfg.model))
    model = MTTFNet(
        n_features=dataset.X.shape[-1],
        n_crack=n_crack,
        n_rebar=n_rebar,
        target_indices=dataset.target_indices,
        **model_config,
    )
    model.load_state_dict(checkpoint["model_state"])
    y_pred = y_scaler.inverse_transform(predict(model, X_test, batch_size=cfg.optim.batch_size))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    grouped_metrics(y_test_raw, y_pred, dataset.target_columns).to_csv(args.output_dir / "metrics_summary.csv", index=False)
    per_target_metrics(y_test_raw, y_pred, dataset.target_columns).to_csv(args.output_dir / "metrics_per_target.csv", index=False)
    pd.DataFrame(y_pred, columns=[f"pred_{c}" for c in dataset.target_columns]).to_csv(args.output_dir / "predictions.csv", index=False)


if __name__ == "__main__":
    main()

