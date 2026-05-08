"""Train MTFF-Net on prepared supervised arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mtff_net.config import load_config
from mtff_net.data import chronological_split, load_dataset
from mtff_net.metrics import grouped_metrics, per_target_metrics
from mtff_net.models import MTTFNet
from mtff_net.train import fit_scalers, predict, seed_everything, train_model, transform_X


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("outputs/dataset/dataset_arrays.npz"))
    parser.add_argument("--metadata", type=Path, default=Path("outputs/dataset/dataset_metadata.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiment"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset)
    train_sl, val_sl, test_sl = chronological_split(len(dataset.X), cfg.dataset.train_ratio, cfg.dataset.val_ratio)
    X_train_raw, X_val_raw, X_test_raw = dataset.X[train_sl], dataset.X[val_sl], dataset.X[test_sl]
    y_train_raw, y_val_raw, y_test_raw = dataset.y[train_sl], dataset.y[val_sl], dataset.y[test_sl]

    scalers = fit_scalers(X_train_raw, y_train_raw)
    X_train = transform_X(X_train_raw, scalers.x_scaler)
    X_val = transform_X(X_val_raw, scalers.x_scaler)
    X_test = transform_X(X_test_raw, scalers.x_scaler)
    y_train = scalers.y_scaler.transform(y_train_raw).astype(np.float32)
    y_val = scalers.y_scaler.transform(y_val_raw).astype(np.float32)

    n_crack = sum(col.startswith("crack_") for col in dataset.target_columns)
    n_rebar = sum(col.startswith("rebar_") for col in dataset.target_columns)
    model_config = {
        "d_model": cfg.model.d_model,
        "n_heads": cfg.model.n_heads,
        "n_layers": cfg.model.n_layers,
        "dropout": cfg.model.dropout,
        "max_len": max(cfg.model.max_len, dataset.X.shape[1]),
        "residual_forecast": cfg.model.residual_forecast,
    }
    model = MTTFNet(
        n_features=dataset.X.shape[-1],
        n_crack=n_crack,
        n_rebar=n_rebar,
        target_indices=dataset.target_indices,
        **model_config,
    )
    trained, history = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        dataset.target_indices,
        epochs=cfg.optim.epochs,
        batch_size=cfg.optim.batch_size,
        learning_rate=cfg.optim.learning_rate,
        weight_decay=cfg.optim.weight_decay,
        patience=cfg.optim.patience,
        smooth_weight=cfg.optim.smooth_weight,
        grad_clip=cfg.optim.grad_clip,
    )

    pred_scaled = predict(trained, X_test, batch_size=cfg.optim.batch_size)
    y_pred = scalers.y_scaler.inverse_transform(pred_scaled)
    metrics = grouped_metrics(y_test_raw, y_pred, dataset.target_columns)
    per_target = per_target_metrics(y_test_raw, y_pred, dataset.target_columns)

    torch.save(
        {
            "model_state": trained.state_dict(),
            "feature_columns": dataset.feature_columns,
            "target_columns": dataset.target_columns,
            "target_indices": dataset.target_indices,
            "model_config": model_config,
        },
        args.output_dir / "mttf_net.pt",
    )
    metrics.to_csv(args.output_dir / "metrics_summary.csv", index=False)
    per_target.to_csv(args.output_dir / "metrics_per_target.csv", index=False)
    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
    pred_df = pd.DataFrame(y_pred, columns=[f"pred_{col}" for col in dataset.target_columns])
    true_df = pd.DataFrame(y_test_raw, columns=[f"true_{col}" for col in dataset.target_columns])
    pd.concat([true_df, pred_df], axis=1).to_csv(args.output_dir / "test_predictions.csv", index=False)

    report = {
        "dataset_metadata": str(args.metadata),
        "checkpoint": str(args.output_dir / "mttf_net.pt"),
        "split": {"train": len(X_train), "validation": len(X_val), "test": len(X_test_raw)},
        "metrics": metrics.to_dict(orient="records"),
    }
    (args.output_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(metrics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
