import numpy as np

from conftest import GRAVITY, synthetic_circle, synthetic_stationary


def test_circle_radius_recovered_from_gt_positions():
    fixture = synthetic_circle()
    radii = np.linalg.norm(fixture.gt_pose["position_enu"][:, :2], axis=1)
    assert np.max(np.abs(radii - 50.0)) < 1e-12


def test_stationary_fixture_has_near_zero_velocity():
    fixture = synthetic_stationary()
    assert np.max(np.abs(fixture.gt_pose["velocity_enu"])) < 1e-12


def test_noiseless_gyro_matches_turn_rate():
    fixture = synthetic_circle()
    expected = 10.0 / 50.0
    assert np.max(np.abs(np.linalg.norm(fixture.gyro, axis=1) - expected)) < 1e-6


def test_noiseless_accel_matches_centripetal_plus_gravity():
    fixture = synthetic_circle()
    # ENU body model is [0, v²/r, +g], so ||a|| = sqrt((v²/r)² + g²).
    expected = np.sqrt((10.0**2 / 50.0) ** 2 + GRAVITY**2)
    assert np.max(np.abs(np.linalg.norm(fixture.accel, axis=1) - expected)) < 1e-12

