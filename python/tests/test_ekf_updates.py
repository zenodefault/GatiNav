import numpy as np

from python.ekf.ekf import ErrorStateEKF


def test_static_propagation_with_gravity_compensation():
    ekf = ErrorStateEKF(P0=np.eye(15), noise_variances=np.zeros(4))
    for _ in range(1000):
        ekf.predict(np.zeros(3), [0.0, 0.0, 9.80665], 0.01)
    assert np.array_equal(ekf.position, np.zeros(3))
    assert np.array_equal(ekf.velocity, np.zeros(3))
    assert np.array_equal(ekf.rotation, np.eye(3))


def test_gnss_update_reduces_position_error_and_covariance():
    ekf = ErrorStateEKF(position=[10.0, 0.0, 0.0], P0=np.eye(15) * 4.0)
    before = np.linalg.norm(ekf.position)
    covariance_before = ekf.P[6, 6]
    ekf.update_gnss([0.0, 0.0, 0.0], 1.0)
    assert np.linalg.norm(ekf.position) < before
    assert ekf.P[6, 6] < covariance_before


def test_zero_velocity_update_corrects_velocity():
    ekf = ErrorStateEKF(velocity=[2.0, 0.0, 0.0], P0=np.eye(15))
    ekf.inject_zero_velocity([0.1, 0.1, 0.1])
    assert abs(ekf.velocity[0]) < 2.0
