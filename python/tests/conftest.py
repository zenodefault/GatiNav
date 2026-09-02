"""Deterministic synthetic fixtures for known-answer tests."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest


GRAVITY = 9.80665


@dataclass
class SyntheticFixture:
    t: np.ndarray
    gyro: np.ndarray
    accel: np.ndarray
    gnss: dict[str, np.ndarray]
    gt_pose: dict[str, np.ndarray]


def _noise(shape, scale, rng):
    scale = np.asarray(scale, dtype=float)
    if np.all(scale == 0):
        return np.zeros(shape)
    return rng.normal(size=shape) * scale


def synthetic_circle(
    duration=60,
    fs=100,
    radius=50,
    speed=10,
    gyro_bias=0.0,
    accel_bias=0.0,
    gyro_noise=0.0,
    accel_noise=0.0,
    seed=0,
):
    """Generate a constant-speed, counter-clockwise ENU circle."""
    n = int(round(duration * fs))
    t = np.arange(n, dtype=np.float64) / fs
    omega = speed / radius
    theta = omega * t
    position = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), np.zeros(n)))
    velocity = np.column_stack((-speed * np.sin(theta), speed * np.cos(theta), np.zeros(n)))
    inward = np.column_stack((-np.cos(theta), -np.sin(theta), np.zeros(n)))
    centripetal = (speed**2 / radius) * inward

    # Body x is forward/tangent and body y is left/inward for this CCW motion.
    gyro = np.tile([0.0, 0.0, omega], (n, 1))
    accel_body = np.column_stack(
        (
            np.zeros(n),
            np.full(n, speed**2 / radius),
            np.full(n, GRAVITY),
        )
    )
    rng = np.random.default_rng(seed)
    gyro = gyro + np.asarray(gyro_bias) + _noise((n, 3), gyro_noise, rng)
    accel = accel_body + np.asarray(accel_bias) + _noise((n, 3), accel_noise, rng)

    gnss_idx = np.arange(0, n, fs, dtype=int)
    gnss = {
        "t": t[gnss_idx],
        "position_enu": position[gnss_idx],
        "accuracy_m": np.zeros(gnss_idx.size),
    }
    gt_pose = {
        "position_enu": position,
        "velocity_enu": velocity,
        "orientation_yaw_rad": theta + np.pi / 2,
        "acceleration_enu": centripetal + np.array([0.0, 0.0, GRAVITY]),
    }
    return SyntheticFixture(t, gyro, accel, gnss, gt_pose)


def synthetic_stationary(
    duration=60,
    fs=100,
    gyro_noise=0.0,
    accel_noise=0.0,
    seed=0,
):
    """Generate a stationary ENU-aligned body with configurable sensor noise."""
    n = int(round(duration * fs))
    t = np.arange(n, dtype=np.float64) / fs
    rng = np.random.default_rng(seed)
    gyro = _noise((n, 3), gyro_noise, rng)
    accel = np.tile([0.0, 0.0, GRAVITY], (n, 1)) + _noise((n, 3), accel_noise, rng)
    gnss_idx = np.arange(0, n, fs, dtype=int)
    gnss = {
        "t": t[gnss_idx],
        "position_enu": np.zeros((gnss_idx.size, 3)),
        "accuracy_m": np.zeros(gnss_idx.size),
    }
    gt_pose = {
        "position_enu": np.zeros((n, 3)),
        "velocity_enu": np.zeros((n, 3)),
        "orientation_yaw_rad": np.zeros(n),
        "acceleration_enu": np.zeros((n, 3)),
    }
    return SyntheticFixture(t, gyro, accel, gnss, gt_pose)


@pytest.fixture
def circle_fixture():
    return synthetic_circle()


@pytest.fixture
def stationary_fixture():
    return synthetic_stationary()


def plot_fixture(fixture, path="results/synthetic_check.png", estimate=None):
    """Write a compact ground-truth/optional-estimate diagnostic plot."""
    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    gt = fixture.gt_pose["position_enu"]
    ax.plot(gt[:, 0], gt[:, 1], label="ground truth")
    if estimate is not None:
        estimate = np.asarray(estimate)
        ax.plot(estimate[:, 0], estimate[:, 1], label="estimate")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
