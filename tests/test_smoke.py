from __future__ import annotations

import torch

from mtff_net.models import MTTFNet


def test_mttf_net_forward_shape() -> None:
    model = MTTFNet(n_features=6, n_crack=2, n_rebar=3, target_indices=[0, 1, 2, 3, 4])
    x = torch.randn(4, 10, 6)
    y = model(x)
    assert tuple(y.shape) == (4, 5)

