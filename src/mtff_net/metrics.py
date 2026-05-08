"""Evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp))
    denom = float(np.mean(np.abs(yt - np.mean(yt))))
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "NMAE": mae / denom if denom else float("nan")}


def grouped_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_columns: list[str], model_name: str = "MTTF-Net") -> pd.DataFrame:
    groups = {
        "all": list(range(len(target_columns))),
        "crack": [i for i, col in enumerate(target_columns) if col.startswith("crack_")],
        "rebar_stress": [i for i, col in enumerate(target_columns) if col.startswith("rebar_")],
    }
    rows = []
    for group, indices in groups.items():
        if not indices:
            continue
        values = regression_metrics(y_true[:, indices], y_pred[:, indices])
        rows.append({"model": model_name, "group": group, **values, "n_values": int(y_true[:, indices].size)})
    return pd.DataFrame(rows)


def per_target_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_columns: list[str], model_name: str = "MTTF-Net") -> pd.DataFrame:
    rows = []
    for idx, target in enumerate(target_columns):
        values = regression_metrics(y_true[:, idx], y_pred[:, idx])
        rows.append(
            {
                "model": model_name,
                "target": target,
                "group": "crack" if target.startswith("crack_") else "rebar_stress",
                **values,
            }
        )
    return pd.DataFrame(rows)

