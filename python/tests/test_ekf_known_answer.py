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


def test_level_rotation_maps_gravity_onto_plus_z():
    from python.ekf.ekf import level_rotation

    rng = np.random.default_rng(0)
    for _ in range(5):
        # random tilt (no yaw convention assumed)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(0.1, 1.2)
        R = _axis_angle(axis, angle)
        f_body = R.T @ np.array([0.0, 0.0, 9.80665])
        leveled = level_rotation(f_body)
        mapped = leveled @ f_body
        assert np.allclose(mapped, [0.0, 0.0, np.linalg.norm(f_body)],
                           atol=1e-9)


def _axis_angle(axis, angle):
    import numpy as np

    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return (np.eye(3) + np.sin(angle) * K
            + (1 - np.cos(angle)) * (K @ K))


def test_leveled_static_phone_no_drift():
    """A tilted static phone must not drift when attitude is leveled."""
    from python.ekf.ekf import ErrorStateEKF, level_rotation

    rng = np.random.default_rng(1)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    tilt = _axis_angle(axis, 0.6)  # ~34 deg mount tilt
    n = 60 * 100
    f_body = tilt.T @ np.array([0.0, 0.0, 9.80665])
    # zero-noise sensors: clean known-answer for the gravity-residual effect
    accel = np.tile(f_body, (n, 1))
    gyro = np.zeros((n, 3))

    # Without leveling the identity attitude integrates the tilt residual.
    ekf = ErrorStateEKF()
    for i in range(n):
        ekf.predict(gyro[i], accel[i], 1 / 100)
    unleveled_drift = np.linalg.norm(ekf.position)
    assert unleveled_drift > 10.0

    # With gravity leveling the residual cancels and the phone stays put.
    ekf = ErrorStateEKF(rotation=level_rotation(accel[:200].mean(axis=0)))
    for i in range(n):
        ekf.predict(gyro[i], accel[i], 1 / 100)
    assert np.linalg.norm(ekf.position) < 1e-3


def test_gnss_velocity_update_keeps_filter_bounded():
    """GNSS position+velocity updates keep a biased straight run bounded.

    A 0.05 m/s^2 accel bias is weakly observable on a straight line (its 1 s
    velocity signature is 0.05 m/s against ~0.5 m/s measurement noise), so
    the assertions are the honest ones: the filter tracks the truth, the
    bias estimate stays small rather than running away (the pre-fix filter
    produced ~6.3 m/s^2), and the velocity update provides the extra
    observability that position-only fusion lacks.
    """
    from python.ekf.ekf import ErrorStateEKF

    fs = 100
    duration = 120
    n = duration * fs
    t = np.arange(n, dtype=np.float64) / fs
    speed = 10.0
    accel_bias = np.array([0.05, 0.0, 0.0])  # m/s^2, body == ENU aligned
    gt_pos = np.column_stack((speed * t, np.zeros(n), np.zeros(n)))
    gt_vel = np.column_stack((np.full(n, speed), np.zeros(n), np.zeros(n)))
    for seed in range(4):
        rng = np.random.default_rng(seed)
        gyro = rng.normal(size=(n, 3)) * 1e-3
        accel = np.tile([0.0, 0.0, 9.80665], (n, 1)) + accel_bias + \
            rng.normal(size=(n, 3)) * 0.01
        ekf = ErrorStateEKF(position=gt_pos[0], velocity=gt_vel[0])
        for i in range(1, n):
            ekf.predict(gyro[i], accel[i], 1 / fs)
            if i % fs == 0:  # 1 Hz GNSS
                ekf.update_gnss(gt_pos[i] + rng.normal(size=3) * 1.5, 4.0)
                ekf.update_gnss_velocity(
                    gt_vel[i] + rng.normal(size=3) * 0.5, 0.5)
        assert np.linalg.norm(ekf.velocity - gt_vel[-1]) < 1.0
        assert np.linalg.norm(ekf.position - gt_pos[-1]) < 5.0
        assert abs(ekf.accel_bias[0]) < 0.1  # no runaway


def test_mixed_gnss_zupt_well_conditioned():
    """Regression: zero process noise drove P singular on Vw16b; defaults
    must keep the update numerically healthy through mixed GNSS+ZUPT runs."""
    from python.ekf.ekf import DEFAULT_NOISE_VARIANCES, ErrorStateEKF

    assert np.all(DEFAULT_NOISE_VARIANCES > 0)
    rng = np.random.default_rng(3)
    ekf = ErrorStateEKF(position=[0.0, 0.0, 0.0], velocity=[10.0, 0.0, 0.0])
    for i in range(2000):
        gyro = rng.normal(size=3) * 0.02
        accel = np.array([0.0, 0.0, 9.80665]) + rng.normal(size=3) * 0.2
        ekf.predict(gyro, accel, 0.01)
        if i % 5 == 0:
            ekf.update_gnss([i * 0.1, 0.0, 0.0], 3.0)
            ekf.update_gnss_velocity([10.0, 0.0, 0.0], 2.0)
        if i % 7 == 0:
            ekf.update_zupt()
        P = ekf.P
        assert np.all(np.isfinite(P))
        assert np.all(np.linalg.eigvalsh(P) > -1e-6)
