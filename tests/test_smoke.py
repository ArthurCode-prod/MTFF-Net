from __future__ import annotations

import torch
import numpy as np

from mtff_net.models import MTTFNet
from mtff_net.spatial import SpatialPair, append_spatial_context_features


def test_mttf_net_forward_shape() -> None:
    model = MTTFNet(n_features=6, n_crack=2, n_rebar=3, target_indices=[0, 1, 2, 3, 4])
    x = torch.randn(4, 10, 6)
    y = model(x)
    assert tuple(y.shape) == (4, 5)


def test_spatial_context_features_shape() -> None:
    feature_columns = ["crack_J1", "crack_J2", "rebar_R1", "rebar_R2", "water_level"]
    crack_columns = ["crack_J1", "crack_J2"]
    rebar_columns = ["rebar_R1", "rebar_R2"]
    pairs = [
        SpatialPair("crack_J1", "rebar_R1", 10.0),
        SpatialPair("crack_J2", "rebar_R1", 5.0),
        SpatialPair("crack_J2", "rebar_R2", 15.0),
    ]
    X = np.ones((3, 4, len(feature_columns)), dtype=np.float32)
    augmented, augmented_columns = append_spatial_context_features(X, feature_columns, crack_columns, rebar_columns, pairs)
    assert augmented.shape == (3, 4, 9)
    assert augmented_columns[-4:] == [
        "spatial_rebar_for_J1",
        "spatial_rebar_for_J2",
        "spatial_crack_for_R1",
        "spatial_crack_for_R2",
    ]
