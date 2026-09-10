"""Unit tests for python/ekf/ekf_denoised.py (Blueprint Phase 1.3)."""

import numpy as np
import pytest
import torch

from python.ekf.ekf_denoised import DenoisedEKF
from python.ml.denoise.model import ResidualTCN


def _identity_denoiser():
    """A no-op denoiser: predicts residual 0 (pass-through)."""
    model = ResidualTCN()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    return model


def test_denoised_ekf_requires_static_init():
    ekf = DenoisedEKF(model=_identity_denoiser())
    with pytest.raises(RuntimeError):
        ekf.predict(np.zeros(3), np.array([0.0, 0.0, 9.80665]), None, 0.01)


def test_denoised_ekf_stationary_stays_finite():
    ekf = DenoisedEKF(model=_identity_denoiser())
    accel = np.tile([0.0, 0.0, 9.80665], (20, 1))
    mag = np.tile([0.0, 1.0, 0.0], (20, 1))
    ekf.init_static(accel, mag)
    for _ in range(200):
        ekf.predict(np.zeros(3), np.array([0.0, 0.0, 9.80665]),
                    np.array([0.0, 1.0, 0.0]), 0.01)
    assert np.all(np.isfinite(ekf.ekf.position))
    assert np.all(np.isfinite(ekf.ekf.P))
    # no sustained motion from a stationary input
    assert np.linalg.norm(ekf.ekf.velocity) < 1.0


def test_denoised_ekf_matches_plain_ekf_with_zero_residual():
    """With a no-op denoiser the propagation equals the baseline EKF."""
    ekf = DenoisedEKF(model=_identity_denoiser())
    accel = np.tile([0.0, 0.0, 9.80665], (20, 1))
    mag = np.tile([0.0, 1.0, 0.0], (20, 1))
    ekf.init_static(accel, mag)
    from python.ekf.ekf import ErrorStateEKF
    plain = ErrorStateEKF()
    for i in range(50):
        gyro = np.zeros(3)
        a = np.array([0.5, 0.0, 9.80665])
        ekf.predict(gyro, a, np.array([0.0, 1.0, 0.0]), 0.01)
        plain.predict(gyro, a, 0.01)
    assert np.allclose(ekf.ekf.position, plain.position, atol=1e-6)


def test_denoised_ekf_static_init_validation():
    ekf = DenoisedEKF()
    with pytest.raises(ValueError):
        ekf.init_static(np.zeros((4, 3)), np.zeros((4, 3)))  # too short


def test_delegates_update_zupt():
    ekf = DenoisedEKF(model=_identity_denoiser())
    accel = np.tile([0.0, 0.0, 9.80665], (20, 1))
    mag = np.tile([0.0, 1.0, 0.0], (20, 1))
    ekf.init_static(accel, mag)
    ekf.ekf.velocity = np.array([2.0, 0.0, 0.0])
    before = ekf.ekf.velocity.copy()
    ekf.update_zupt()
    assert np.linalg.norm(ekf.ekf.velocity) < np.linalg.norm(before)