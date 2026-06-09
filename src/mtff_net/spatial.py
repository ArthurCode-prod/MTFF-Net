"""Utilities for distance-informed crack--rebar context features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SpatialPair:
    """Plan-based association between one crack gauge and one rebar-stress gauge."""

    crack_column: str
    rebar_column: str
    distance_m: float


def _column_indices(columns: list[str], selected: Iterable[str]) -> list[int]:
    lookup = {name: idx for idx, name in enumerate(columns)}
    missing = [name for name in selected if name not in lookup]
    if missing:
        raise ValueError(f"Columns not found in feature_columns: {missing}")
    return [lookup[name] for name in selected]


def _remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix) :] if text.startswith(prefix) else text


def build_inverse_distance_weights(
    pairs: Iterable[SpatialPair],
    crack_columns: list[str],
    rebar_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build row- and column-normalized inverse-distance weights.

    The returned matrices have shape ``(n_crack, n_rebar)``. Row-normalized
    weights summarize rebar context for each crack gauge, while column-normalized
    weights summarize crack context for each rebar-stress gauge.
    """

    pair_list = list(pairs)
    crack_lookup = {name: idx for idx, name in enumerate(crack_columns)}
    rebar_lookup = {name: idx for idx, name in enumerate(rebar_columns)}
    scores = np.zeros((len(crack_columns), len(rebar_columns)), dtype=np.float64)

    for pair in pair_list:
        if pair.distance_m <= 0:
            raise ValueError("SpatialPair.distance_m must be positive.")
        if pair.crack_column not in crack_lookup:
            raise ValueError(f"Unknown crack column: {pair.crack_column}")
        if pair.rebar_column not in rebar_lookup:
            raise ValueError(f"Unknown rebar column: {pair.rebar_column}")
        i = crack_lookup[pair.crack_column]
        j = rebar_lookup[pair.rebar_column]
        scores[i, j] = 1.0 / pair.distance_m

    row_sum = scores.sum(axis=1, keepdims=True)
    col_sum = scores.sum(axis=0, keepdims=True)
    row_weights = np.divide(scores, row_sum, out=np.zeros_like(scores), where=row_sum > 0)
    col_weights = np.divide(scores, col_sum, out=np.zeros_like(scores), where=col_sum > 0)
    return row_weights.astype(np.float32), col_weights.astype(np.float32)


def append_spatial_context_features(
    X: np.ndarray,
    feature_columns: list[str],
    crack_columns: list[str],
    rebar_columns: list[str],
    pairs: Iterable[SpatialPair],
) -> tuple[np.ndarray, list[str]]:
    """Append distance-weighted context features to supervised sequence arrays.

    Parameters
    ----------
    X:
        Supervised input sequence array with shape ``(n_samples, window, n_features)``.
    feature_columns:
        Names corresponding to the last axis of ``X``.
    crack_columns, rebar_columns:
        Target response columns used to form the spatial context.
    pairs:
        Plan-based crack--rebar associations and distances.
    """

    if X.ndim != 3:
        raise ValueError("X must have shape (n_samples, window, n_features).")
    crack_indices = _column_indices(feature_columns, crack_columns)
    rebar_indices = _column_indices(feature_columns, rebar_columns)
    row_weights, col_weights = build_inverse_distance_weights(pairs, crack_columns, rebar_columns)

    crack_history = X[:, :, crack_indices]
    rebar_history = X[:, :, rebar_indices]
    rebar_context = np.einsum("nlj,ij->nli", rebar_history, row_weights).astype(np.float32)
    crack_context = np.einsum("nli,ij->nlj", crack_history, col_weights).astype(np.float32)

    spatial_columns = [f"spatial_rebar_for_{_remove_prefix(name, 'crack_')}" for name in crack_columns]
    spatial_columns += [f"spatial_crack_for_{_remove_prefix(name, 'rebar_')}" for name in rebar_columns]
    augmented = np.concatenate([X.astype(np.float32, copy=False), rebar_context, crack_context], axis=2)
    return augmented, feature_columns + spatial_columns
