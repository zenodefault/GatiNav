"""Tests for specs/01 section 7 preprocessing (gravity removal, windows)."""

import numpy as np
import pytest

from conftest import synthetic_circle, synthetic_stationary
from python.io.iovnb_loader import IOVNBDSample
from python.io.preprocessing import gravity_remove, window_segmentation


def _as_sample(fx):
    """Adapt a SyntheticFixture to an IOVNBDSample for windowing tests."""
    n = fx.t.size
    gnss = np.zeros(n, dtype=[("t", "f8"), ("lat_deg", "f8"), ("lon_deg", "f8"),
                              ("alt_m", "f8"), ("accuracy_m", "f8")])
    gnss["t"] = fx.t
    return IOVNBDSample(t=fx.t, gyro=fx.gyro, accel=fx.accel,
                        mag=np.ones((n, 3)), gnss=gnss, gt_pose=np.zeros(
                            n, dtype=[("t", "f8"), ("x", "f8")]))


def test_gravity_remove_stationary():
    # Stationary ENU-aligned body: linear acceleration must be ~0 after
    # gravity removal (specs/01 §8.2, tolerance VERIFY = 1e-6).
    fx = synthetic_stationary(duration=60, fs=100)
    linear = gravity_remove(fx.accel, fs=100.0)
    assert linear.shape == fx.accel.shape
    assert linear.dtype == np.float64
    assert np.max(np.abs(linear)) < 1e-6


def test_gravity_remove_circle_preserves_centripetal():
    # Constant circle: body-Z gravity is removed, the centripetal component
    # (v^2/r = 100/50 = 2 m/s^2 on body Y) is preserved untouched.
    fx = synthetic_circle()
    linear = gravity_remove(fx.accel, fs=100.0)
    assert np.abs(linear[:, 1].mean() - 100.0 / 50.0) < 1e-6
    assert np.max(np.abs(linear[:, 0])) < 1e-6
    assert np.max(np.abs(linear[:, 2])) < 1e-6


def test_gravity_remove_validation():
    fx = synthetic_stationary(duration=1, fs=100)
    with pytest.raises(ValueError):
        gravity_remove(fx.accel, fs=100.0, gravity_cutoff_hz=0.0)
    with pytest.raises(ValueError):
        gravity_remove(fx.accel, fs=100.0, gravity_cutoff_hz=60.0)
    with pytest.raises(ValueError):
        gravity_remove(fx.accel[:, :2], fs=100.0)


def test_window_count_and_overlap():
    # 60 s at 100 Hz, 5 s windows with 2.5 s stride:
    #   n_win = 500, n_stride = 250, count = 1 + (6000-500)//250 = 23,
    #   consecutive windows overlap by n_win - n_stride = 250 samples.
    fx = synthetic_circle(duration=60, fs=100)
    stream = _as_sample(fx)
    windows = window_segmentation(stream, window_s=5.0, stride_s=2.5)
    n_win, n_stride = 500, 250
    assert len(windows) == 1 + (fx.t.size - n_win) // n_stride
    for k, w in enumerate(windows):
        assert w.t.size == n_win
        assert w.t[0] == pytest.approx(fx.t[k * n_stride])
        assert w.t[-1] == pytest.approx(fx.t[k * n_stride + n_win - 1])
        assert w.gyro.shape == (n_win, 3)
        assert w.accel.shape == (n_win, 3)
        assert w.gnss.shape[0] == n_win
        assert w.gt_pose.shape[0] == n_win
    overlap = n_win - n_stride
    assert np.allclose(windows[0].gyro[overlap:], windows[1].gyro[:n_stride])
    # last window ends exactly at the stream end (complete windows only)
    assert windows[-1].t[-1] == pytest.approx(fx.t[-1])


def test_window_parameter_validation():
    fx = synthetic_circle(duration=10, fs=100)
    stream = _as_sample(fx)
    with pytest.raises(ValueError):
        window_segmentation(stream, window_s=4.0, stride_s=1.0)
    with pytest.raises(ValueError):
        window_segmentation(stream, window_s=11.0, stride_s=1.0)
    with pytest.raises(ValueError):
        window_segmentation(stream, window_s=5.0, stride_s=0.0)