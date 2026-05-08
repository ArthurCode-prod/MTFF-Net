"""Configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    datetime_column: str = "datetime"
    target_prefixes: list[str] = field(default_factory=lambda: ["crack_", "rebar_"])
    feature_prefixes: list[str] = field(default_factory=lambda: ["crack_", "rebar_", "temperature_", "environment_"])
    extra_feature_columns: list[str] = field(
        default_factory=lambda: [
            "water_level",
            "rainfall_8h",
            "rainfall_24h",
            "rainfall_72h",
            "day_sin",
            "day_cos",
            "hour_sin",
            "hour_cos",
        ]
    )
    window: int = 21
    horizon: int = 6
    train_ratio: float = 0.70
    val_ratio: float = 0.15


@dataclass
class ModelConfig:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.15
    max_len: int = 128
    residual_forecast: bool = True


@dataclass
class OptimConfig:
    epochs: int = 130
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 12
    smooth_weight: float = 0.005
    grad_clip: float = 1.0


@dataclass
class ExperimentConfig:
    seed: int = 42
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    config = ExperimentConfig()
    if path is None:
        return config
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return _merge_dataclass(config, values)

