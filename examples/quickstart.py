"""Minimal MTFF-Net smoke test with toy data.

This example is intentionally synthetic. It checks that the data pipeline and
model can run, but it is not connected to the manuscript experiments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mtff_net.data import build_supervised_dataset, chronological_split
from mtff_net.metrics import grouped_metrics
from mtff_net.models import MTTFNet
from mtff_net.train import fit_scalers, predict, seed_everything, train_model, transform_X


def make_toy_monitoring_table(n_steps: int = 96) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    time = pd.date_range("2024-01-01", periods=n_steps, freq="8h")
    phase = np.linspace(0, 5 * np.pi, n_steps)
    water = 14.2 + 0.25 * np.sin(phase / 2)
    rain24 = np.maximum(0, 0.5 + 0.3 * np.sin(phase + 0.8))
    temp = 10 + 6 * np.sin(phase)
    crack = 0.10 + 0.006 * np.sin(phase / 1.7) + 0.0007 * rain24 + rng.normal(0, 0.0007, n_steps)
    stress = 120 + 1.2 * np.sin(phase / 1.5) + 0.08 * (water - water.mean()) + rng.normal(0, 0.12, n_steps)
    return pd.DataFrame(
        {
            "datetime": time,
            "crack_J01": crack,
            "rebar_R01": stress,
            "temperature_J01": temp,
            "temperature_R01": temp + rng.normal(0, 0.3, n_steps),
            "water_level": water,
            "rainfall_8h": rain24 / 3,
            "rainfall_24h": rain24,
            "rainfall_72h": rain24 * 2.5,
            "environment_aux_01": temp - 0.4,
        }
    )


def main() -> None:
    seed_everything(42)
    dataset = build_supervised_dataset(make_toy_monitoring_table(), window=8, horizon=2)
    train_sl, val_sl, test_sl = chronological_split(len(dataset.X), train_ratio=0.70, val_ratio=0.15)

    X_train_raw, X_val_raw, X_test_raw = dataset.X[train_sl], dataset.X[val_sl], dataset.X[test_sl]
    y_train_raw, y_val_raw, y_test_raw = dataset.y[train_sl], dataset.y[val_sl], dataset.y[test_sl]
    scalers = fit_scalers(X_train_raw, y_train_raw)
    X_train = transform_X(X_train_raw, scalers.x_scaler)
    X_val = transform_X(X_val_raw, scalers.x_scaler)
    X_test = transform_X(X_test_raw, scalers.x_scaler)
    y_train = scalers.y_scaler.transform(y_train_raw).astype(np.float32)
    y_val = scalers.y_scaler.transform(y_val_raw).astype(np.float32)

    model = MTTFNet(
        n_features=dataset.X.shape[-1],
        n_crack=1,
        n_rebar=1,
        target_indices=dataset.target_indices,
        d_model=24,
        n_heads=4,
        n_layers=1,
        dropout=0.10,
        max_len=16,
    )
    trained, history = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        dataset.target_indices,
        epochs=5,
        batch_size=16,
        patience=3,
    )
    y_pred = scalers.y_scaler.inverse_transform(predict(trained, X_test))
    print(grouped_metrics(y_test_raw, y_pred, dataset.target_columns).round(4).to_string(index=False))
    print(f"Completed toy smoke test in {len(history)} epochs.")


if __name__ == "__main__":
    main()
