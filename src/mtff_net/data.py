"""Data loading and supervised sequence construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SupervisedDataset:
    X: np.ndarray
    y: np.ndarray
    times: np.ndarray
    feature_columns: list[str]
    target_columns: list[str]
    target_indices: list[int]
    window: int
    horizon: int


def parse_datetime(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed


def add_calendar_features(df: pd.DataFrame, datetime_column: str = "datetime") -> pd.DataFrame:
    out = df.copy()
    dt = parse_datetime(out[datetime_column])
    day_angle = 2.0 * np.pi * dt.dt.dayofyear.to_numpy() / 366.0
    hour_angle = 2.0 * np.pi * (dt.dt.hour.to_numpy() + dt.dt.minute.to_numpy() / 60.0) / 24.0
    out["day_sin"] = np.sin(day_angle)
    out["day_cos"] = np.cos(day_angle)
    out["hour_sin"] = np.sin(hour_angle)
    out["hour_cos"] = np.cos(hour_angle)
    return out


def read_database_table(database_url: str, query: str) -> pd.DataFrame:
    try:
        from sqlalchemy import create_engine
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install the optional database dependencies with `pip install -e .[db]`.") from exc

    engine = create_engine(database_url)
    with engine.begin() as connection:
        return pd.read_sql_query(query, connection)


def infer_columns(
    df: pd.DataFrame,
    target_prefixes: Iterable[str] = ("crack_", "rebar_"),
    feature_prefixes: Iterable[str] = ("crack_", "rebar_", "temperature_", "environment_"),
    extra_feature_columns: Iterable[str] = (
        "water_level",
        "rainfall_8h",
        "rainfall_24h",
        "rainfall_72h",
        "day_sin",
        "day_cos",
        "hour_sin",
        "hour_cos",
    ),
) -> tuple[list[str], list[str]]:
    target_prefixes = tuple(target_prefixes)
    feature_prefixes = tuple(feature_prefixes)
    target_columns = [col for col in df.columns if col.startswith(target_prefixes)]
    prefix_features = [col for col in df.columns if col.startswith(feature_prefixes)]
    extra_features = [col for col in extra_feature_columns if col in df.columns]
    feature_columns = list(dict.fromkeys(prefix_features + extra_features))
    if not target_columns:
        raise ValueError("No target columns were found. Expected columns beginning with crack_ or rebar_.")
    if not feature_columns:
        raise ValueError("No feature columns were found.")
    missing_targets = [col for col in target_columns if col not in feature_columns]
    feature_columns.extend(missing_targets)
    return feature_columns, target_columns


def make_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    window: int,
    horizon: int,
    datetime_column: str = "datetime",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if window < 1 or horizon < 1:
        raise ValueError("window and horizon must be positive integers.")
    features = df[feature_columns].to_numpy(dtype=np.float32)
    targets = df[target_columns].to_numpy(dtype=np.float32)
    times = parse_datetime(df[datetime_column]).to_numpy()

    X, y, t = [], [], []
    last_start = len(df) - window - horizon + 1
    if last_start <= 0:
        raise ValueError("The aligned table is too short for the selected window and horizon.")
    for start in range(last_start):
        end = start + window
        target_index = end + horizon - 1
        X.append(features[start:end])
        y.append(targets[target_index])
        t.append(times[target_index])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), np.asarray(t)


def build_supervised_dataset(
    df: pd.DataFrame,
    window: int = 21,
    horizon: int = 6,
    datetime_column: str = "datetime",
    target_prefixes: Iterable[str] = ("crack_", "rebar_"),
    feature_prefixes: Iterable[str] = ("crack_", "rebar_", "temperature_", "environment_"),
    extra_feature_columns: Iterable[str] = (
        "water_level",
        "rainfall_8h",
        "rainfall_24h",
        "rainfall_72h",
        "day_sin",
        "day_cos",
        "hour_sin",
        "hour_cos",
    ),
) -> SupervisedDataset:
    aligned = df.copy()
    aligned[datetime_column] = parse_datetime(aligned[datetime_column])
    aligned = aligned.dropna(subset=[datetime_column]).sort_values(datetime_column).drop_duplicates(datetime_column)
    aligned = add_calendar_features(aligned, datetime_column=datetime_column)
    feature_columns, target_columns = infer_columns(aligned, target_prefixes, feature_prefixes, extra_feature_columns)
    aligned = aligned.dropna(subset=feature_columns + target_columns).reset_index(drop=True)
    X, y, times = make_sequences(aligned, feature_columns, target_columns, window, horizon, datetime_column)
    target_indices = [feature_columns.index(col) for col in target_columns]
    return SupervisedDataset(X, y, times, feature_columns, target_columns, target_indices, window, horizon)


def chronological_split(n_items: int, train_ratio: float = 0.70, val_ratio: float = 0.15) -> tuple[slice, slice, slice]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be smaller than 1.")
    n_train = int(n_items * train_ratio)
    n_val = int(n_items * val_ratio)
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n_items)


def save_dataset(dataset: SupervisedDataset, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "dataset_arrays.npz",
        X=dataset.X,
        y=dataset.y,
        times=dataset.times.astype("datetime64[ns]").astype("int64"),
        feature_columns=np.asarray(dataset.feature_columns),
        target_columns=np.asarray(dataset.target_columns),
        target_indices=np.asarray(dataset.target_indices),
        window=np.asarray([dataset.window]),
        horizon=np.asarray([dataset.horizon]),
    )


def load_dataset(arrays_path: str | Path) -> SupervisedDataset:
    arrays = np.load(arrays_path, allow_pickle=True)
    return SupervisedDataset(
        X=arrays["X"].astype(np.float32),
        y=arrays["y"].astype(np.float32),
        times=pd.to_datetime(arrays["times"].astype("int64")).to_numpy(),
        feature_columns=arrays["feature_columns"].astype(str).tolist(),
        target_columns=arrays["target_columns"].astype(str).tolist(),
        target_indices=arrays["target_indices"].astype(int).tolist(),
        window=int(arrays["window"][0]),
        horizon=int(arrays["horizon"][0]),
    )

