import numpy as np
import pytest

from conftest import synthetic_circle, synthetic_stationary


def test_zero_noise_recovers():
    from python.ekf.ekf import ErrorStateEKF

    fixture = synthetic_circle()
    yaw = np.pi / 2
    initial_rotation = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0.0],
         [np.sin(yaw), np.cos(yaw), 0.0],
         [0.0, 0.0, 1.0]]
    )
    ekf = ErrorStateEKF(
        rotation=initial_rotation,
        position=fixture.gt_pose["position_enu"][0],
        velocity=fixture.gt_pose["velocity_enu"][0],
    )
    for i in range(len(fixture.t) - 1):
        ekf.predict(fixture.gyro[i], fixture.accel[i], 1 / 100)
    error = np.linalg.norm(ekf.position - fixture.gt_pose["position_enu"][-1])
    assert error < 0.5


def test_bias_is_estimated():
    from python.ekf.ekf import ErrorStateEKF

    fixture = synthetic_circle(gyro_bias=[0.0, 0.0, 0.01])
    yaw = np.pi / 2
    initial_rotation = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0.0],
         [np.sin(yaw), np.cos(yaw), 0.0],
         [0.0, 0.0, 1.0]]
    )
    ekf = ErrorStateEKF(
        rotation=initial_rotation,
        position=fixture.gt_pose["position_enu"][0],
        velocity=fixture.gt_pose["velocity_enu"][0],
    )
    for i in range(len(fixture.t) - 1):
        ekf.predict(fixture.gyro[i], fixture.accel[i], 1 / 100)
    ekf.inject_zero_velocity([1.0, 1.0, 1.0])
    assert abs(ekf.gyro_bias[2]) < 0.01


def test_zupt_stationary():
    from python.ekf.ekf import ErrorStateEKF

    fixture = synthetic_stationary()
    ekf = ErrorStateEKF()
    for i in range(len(fixture.t)):
        ekf.predict(fixture.gyro[i], fixture.accel[i], 1 / 100)
        ekf.update_zupt()
    drift = np.linalg.norm(ekf.position - fixture.gt_pose["position_enu"][-1])
    assert drift < 1.0
