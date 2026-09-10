"""Unit tests for python/calibration/ahrs.py (Blueprint Phase 1.1)."""

import numpy as np
import pytest

from python.calibration.ahrs import Ahrs
from python.ekf.rotations import rotation_matrix_to_quaternion


def _rotate(vectors, rotation):
    """Map level-frame vectors into the body frame (R^T @ v)."""
    return np.asarray(vectors) @ rotation


def test_init_static_recovers_known_rotation():
    """A known mount rotation must be recovered from gravity + mag."""
    rng = np.random.default_rng(0)
    # body -> level rotation: 10 deg pitch about y, 20 deg roll about x,
    # 30 deg yaw about z (arbitrary but known)
    roll, pitch, yaw = np.deg2rad([20.0, 10.0, 30.0])
    rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    rotation = rz @ ry @ rx  # body -> level

    n = 100
    accel = _rotate(np.tile([0.0, 0.0, 9.80665], (n, 1)), rotation)
    mag = _rotate(np.tile([0.0, 1.0, 0.0], (n, 1)), rotation)
    accel += rng.normal(0, 0.01, accel.shape)
    mag += rng.normal(0, 0.001, mag.shape)

    ahrs = Ahrs().init_static(accel, mag)
    recovered = ahrs.rotation
    # recovered @ rotation.T should be close to identity
    residual = recovered @ rotation.T
    assert np.allclose(residual, np.eye(3), atol=1e-2), residual


def test_init_static_rejects_vertical_field():
    ahrs = Ahrs()
    accel = np.tile([0.0, 0.0, 9.80665], (10, 1))
    mag = np.tile([0.0, 0.0, 1.0], (10, 1))  # no horizontal component
    with pytest.raises(ValueError):
        ahrs.init_static(accel, mag)


def test_update_keeps_attitude_stable_on_stationary_noise():
    ahrs = Ahrs().init_static(
        np.tile([0.0, 0.0, 9.80665], (20, 1)),
        np.tile([0.0, 1.0, 0.0], (20, 1)))
    r0 = ahrs.rotation.copy()
    rng = np.random.default_rng(1)
    for _ in range(100):
        gyro = rng.normal(0, 1e-3, 3)
        accel = np.array([0.0, 0.0, 9.80665]) + rng.normal(0, 0.02, 3)
        ahrs.update(gyro, accel, np.array([0.0, 1.0, 0.0]), 0.01)
    drift = np.linalg.norm(ahrs.rotation - r0)
    assert drift < 0.05  # attitude should stay near its initial value


def test_forward_accel_zero_gravity_leak():
    """Forward-axis projection of a pure-gravity signal must be ~0."""
    ahrs = Ahrs().init_static(
        np.tile([0.0, 0.0, 9.80665], (20, 1)),
        np.tile([0.0, 1.0, 0.0], (20, 1)))
    f = ahrs.forward_accel(np.array([0.0, 0.0, 9.80665]))
    assert abs(f) < 1e-9


def test_update_requires_init():
    ahrs = Ahrs()
    with pytest.raises(RuntimeError):
        ahrs.update(np.zeros(3), np.array([0.0, 0.0, 9.80665]), None, 0.01)


def test_rotation_matrix_valid():
    ahrs = Ahrs().init_static(
        np.tile([0.0, 0.0, 9.80665], (20, 1)),
        np.tile([0.0, 1.0, 0.0], (20, 1)))
    r = ahrs.rotation
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(r), 1.0, atol=1e-9)
    # quaternion round-trip
    q = rotation_matrix_to_quaternion(r)
    from python.ekf.rotations import quaternion_to_rotation_matrix
    assert np.allclose(quaternion_to_rotation_matrix(q), r, atol=1e-9)