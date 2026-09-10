"""Unit tests for python/ml/denoise (Blueprint Phase 1.2)."""

import numpy as np
import pytest
import torch

from python.ml.denoise.dataset import (DenoiseWindows, WINDOW_S, FS,
                                       driver_folds, normalize_stats)
from python.ml.denoise.model import PARAM_COUNT, ResidualTCN


def test_param_count_within_budget():
    assert PARAM_COUNT < 500_000
    assert PARAM_COUNT > 1_000


def test_forward_shape_and_causality():
    w = int(round(WINDOW_S * FS))
    model = ResidualTCN()
    x = torch.randn(3, 1, w)
    out = model(x)
    assert out.shape == (3, 1)
    # causality: the window-end output depends only on its receptive field
    # (dilations 1,2,4,8, kernel 7 -> RF = 7 + 2*6*15 = 187). Perturbing
    # samples older than the receptive field must not change the output.
    rf = 7 + 2 * 6 * (1 + 2 + 4 + 8)
    x2 = x.clone()
    x2[:, :, : w - rf - 5] = -99.0  # perturb only the oldest samples
    assert torch.allclose(model(x), model(x2), atol=1e-5)
    # ...but the last sample is inside the field: changing it must matter
    # (random-init weights attenuate the response geometrically through
    # the blocks, so we only require a non-zero change, not a large one)
    x3 = x.clone()
    x3[:, :, -1] = 99.0
    assert not torch.equal(model(x), model(x3))


def test_backward_trains_mse():
    w = int(round(WINDOW_S * FS))
    model = ResidualTCN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(8, 1, w)
    y = torch.randn(8, 1)
    loss0 = None
    for _ in range(5):
        opt.zero_grad()
        loss = torch.mean((model(x) - y) ** 2)
        loss.backward()
        opt.step()
        loss0 = float(loss)
    assert np.isfinite(loss0)


def test_driver_folds_and_normalization():
    windows = DenoiseWindows(
        x=np.linspace(0.0, 1.0, 12 * 10).reshape(12, 10),
        target=np.zeros(12),
        driver=np.array(["Driver A"] * 3 + ["Driver B"] * 3
                        + ["Driver C"] * 3 + ["Driver D"] * 3),
        segment=np.arange(12).astype(str), t_end=np.arange(12.0))
    folds = driver_folds(windows)
    assert set(folds) == {"Driver A", "Driver B", "Driver C", "Driver D"}
    for driver, fold in folds.items():
        tr, va, te = (fold["train_idx"], fold["val_idx"], fold["test_idx"])
        assert np.intersect1d(tr, te).size == 0
        assert np.intersect1d(va, te).size == 0
    mean, std = normalize_stats(windows.x, folds["Driver A"]["train_idx"])
    assert np.isfinite(mean) and std > 0


def test_denoise_windows_empty_ok():
    windows = DenoiseWindows(np.empty((0, 10)), np.empty(0), np.array([]),
                             np.array([]), np.empty(0))
    assert len(windows) == 0
    with pytest.raises(ValueError):
        driver_folds(windows)