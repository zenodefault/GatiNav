import numpy as np
import pytest

from conftest import synthetic_circle, synthetic_stationary


@pytest.mark.xfail(reason="EKF not implemented yet — Phase 2", strict=False)
def test_zero_noise_recovers():
    from python.ekf.ekf import ErrorStateEKF

    fixture = synthetic_circle()
    ekf = ErrorStateEKF()
    for i in range(len(fixture.t)):
        ekf.predict(fixture.gyro[i], fixture.accel[i], 1 / 100)
    error = np.linalg.norm(ekf.position - fixture.gt_pose["position_enu"][-1])
    assert error < 0.5


@pytest.mark.xfail(reason="EKF not implemented yet — Phase 2", strict=False)
def test_bias_is_estimated():
    from python.ekf.ekf import ErrorStateEKF

    fixture = synthetic_circle(gyro_bias=[0.0, 0.0, 0.01])
    ekf = ErrorStateEKF()
    for i in range(len(fixture.t)):
        ekf.predict(fixture.gyro[i], fixture.accel[i], 1 / 100)
    assert np.isclose(ekf.gyro_bias[2], 0.01, rtol=0.05)


@pytest.mark.xfail(reason="EKF not implemented yet — Phase 2", strict=False)
def test_zupt_stationary():
    from python.ekf.ekf import ErrorStateEKF

    fixture = synthetic_stationary()
    ekf = ErrorStateEKF()
    for i in range(len(fixture.t)):
        ekf.predict(fixture.gyro[i], fixture.accel[i], 1 / 100)
        ekf.update(np.zeros(3), np.eye(3), np.zeros((3, 15)), np.eye(3))
    drift = np.linalg.norm(ekf.position - fixture.gt_pose["position_enu"][-1])
    assert drift < 1.0
