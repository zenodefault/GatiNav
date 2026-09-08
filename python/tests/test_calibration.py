"""Known-answer tests for the Phase 1 CalibrationEngine.

Covers: stationary bias + leveling recovery on a tilted mount, mounting
yaw from GNSS course on straight driving, idle-vibration immunity vs
mount-movement detection, the phone->vehicle frame transform, and
quasi-static window discovery. All fixtures are synthetic with known
answers; ground truth is never consulted by the module under test.
"""

import numpy as np
import pytest

from python.calibration.engine import (CalibrationEngine, calibrate_session,
                                       find_static_window, quasi_static_mask)
from python.io.iovnb_loader import GNSS_DTYPE, GT_POSE_DTYPE, IOVNBDSample

G = 9.80665
_A = 6378137.0
_LAT0, _LON0 = np.deg2rad(12.97), np.deg2rad(77.56)


def _rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _session(t, gyro, accel, north, gnss_every_s=1.0):
    """IOVNBDSample with GNSS at phone rate, fixes every gnss_every_s s.

    Mirrors the loader contract: GNSS rows align 1:1 with IMU rows, and
    non-fix rows carry NaN (as the loader does inside outages).
    """
    fs = 1.0 / float(np.median(np.diff(t)))
    gi = np.arange(0, t.size, int(round(gnss_every_s * fs)))
    gnss = np.zeros(t.size, dtype=GNSS_DTYPE)
    gnss["t"] = t
    gnss["lat_deg"] = np.nan
    gnss["lon_deg"] = np.nan
    gnss["alt_m"] = np.nan
    gnss["accuracy_m"] = np.nan
    gnss["lat_deg"][gi] = np.rad2deg(_LAT0 + north[gi] / _A)
    gnss["lon_deg"][gi] = np.rad2deg(_LON0)
    gnss["alt_m"][gi] = 900.0
    gnss["accuracy_m"][gi] = 2.0
    gt = np.zeros(t.size, dtype=GT_POSE_DTYPE)
    gt["t"] = t
    return IOVNBDSample(t=t, gyro=gyro, accel=accel,
                        mag=np.zeros_like(accel), gnss=gnss, gt_pose=gt)


def test_stationary_calibration_recovers_biases_and_levels_mount():
    fs, n = 100.0, 2000  # 20 s
    t = np.arange(n) / fs
    theta = np.deg2rad(15.0)  # phone pitched 15 deg on its mount
    gyro_bias = np.array([0.004, -0.006, 0.008])
    accel_bias = np.array([0.05, -0.03, 0.0])
    rng = np.random.default_rng(0)
    f_phone = _rot_x(theta) @ np.array([0.0, 0.0, G])
    gyro = np.tile(gyro_bias, (n, 1)) + rng.normal(scale=1e-4, size=(n, 3))
    accel = (np.tile(f_phone + accel_bias, (n, 1))
             + rng.normal(scale=1e-3, size=(n, 3)))

    engine = CalibrationEngine().calibrate_stationary(gyro, accel)

    assert engine.calibrated
    np.testing.assert_allclose(engine.gyro_bias, gyro_bias, atol=1e-3)
    gyro_v, accel_v = engine.to_vehicle_frame(gyro, accel)
    # Tilted mount must level out: rest specific force maps onto +z only.
    np.testing.assert_allclose(np.mean(accel_v, axis=0), [0.0, 0.0, G],
                               atol=0.05)
    np.testing.assert_allclose(np.mean(gyro_v, axis=0), [0.0, 0.0, 0.0],
                               atol=1e-3)


def test_mount_yaw_from_gnss_course_known_answer():
    fs, duration = 100.0, 30.0
    n = int(duration * fs)
    t = np.arange(n) / fs
    speed = 10.0
    north = speed * t
    rng = np.random.default_rng(1)
    gyro = rng.normal(scale=1e-4, size=(n, 3))  # straight driving: ~zero yaw
    accel = np.tile([0.0, 0.0, G], (n, 1)) + rng.normal(scale=1e-3, size=(n, 3))
    session = _session(t, gyro, accel, north)

    engine = CalibrationEngine().calibrate_stationary(gyro, accel)
    engine.calibrate_yaw(t, gyro, session.gnss)

    # Driving due north, phone x-axis level: vehicle course in the leveled
    # phone frame must be pi/2 (ENU north), so forward accel maps to +north.
    assert abs(engine.yaw_mount - np.pi / 2.0) < 0.05
    _, accel_v = engine.to_vehicle_frame(np.zeros((1, 3)),
                                         np.array([[1.0, 0.0, 0.0]]))
    assert accel_v[0, 1] > 0.9 and abs(accel_v[0, 0]) < 0.1


def test_idle_vibration_does_not_trigger_movement():
    fs, n = 100.0, 1000  # 10 s
    t = np.arange(n) / fs
    # Engine-idle vibration: high-frequency, low-amplitude, no net tilt.
    vib = 0.25 * np.sin(2 * np.pi * 30.0 * t)
    gyro = 0.05 * np.sin(2 * np.pi * 25.0 * t)[:, None] * np.ones((1, 3))
    accel = np.tile([0.0, 0.0, G], (n, 1)) + vib[:, None]
    engine = CalibrationEngine().calibrate_stationary(gyro, accel)

    flags = engine.movement_flags(accel, gyro, fs)
    assert not flags.any()


def test_mount_movement_triggered_by_gravity_step():
    fs, n = 100.0, 2000  # 20 s
    t = np.arange(n) / fs
    theta = np.deg2rad(10.0)
    accel = np.tile([0.0, 0.0, G], (n, 1))
    accel[n // 2:] = _rot_x(theta) @ np.array([0.0, 0.0, G])
    gyro = np.zeros((n, 3))
    engine = CalibrationEngine().calibrate_stationary(gyro[:n // 4],
                                                      accel[:n // 4])

    flags = engine.movement_flags(accel, gyro, fs)
    assert not flags[:100].any()          # settled start: no movement
    assert flags[n // 2 + 600:].any()     # after the 10 deg step: flagged


def test_to_vehicle_frame_known_answer():
    engine = CalibrationEngine()
    engine.gyro_bias = np.array([0.1, 0.2, 0.3])
    engine.accel_bias = np.array([0.01, 0.02, 0.03])
    yaw = np.deg2rad(30.0)
    engine.yaw_mount = yaw
    c, s = np.cos(yaw), np.sin(yaw)
    engine.rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    engine.calibrated = True

    gyro = np.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]])
    accel = np.array([[0.01, 0.02, 0.03], [1.0, 2.0, 3.0]])
    gyro_v, accel_v = engine.to_vehicle_frame(gyro, accel)

    np.testing.assert_allclose(gyro_v[0], [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(accel_v[0], [0.0, 0.0, 0.0], atol=1e-12)
    # Bias is subtracted before the rotation: expected = R @ (v - bias).
    np.testing.assert_allclose(accel_v[1],
                               engine.rotation @ np.array([0.99, 1.98, 2.97]),
                               atol=1e-12)


def test_find_static_window_ignores_motion_and_calibrate_session():
    fs, n = 100.0, 3000  # 30 s: 10 s lateral acceleration, 20 s static
    t = np.arange(n) / fs
    rng = np.random.default_rng(2)
    gyro = rng.normal(scale=1e-3, size=(n, 3))  # straight motion: no yaw
    accel = np.tile([0.0, 0.0, G], (n, 1))
    accel[:1000, 1] = 8.0  # sustained lateral force: not rest (rejects 1 s
    # straddling windows too: 1 moving sample => var 0.64 > 0.5)
    accel[1000:, 2] += rng.normal(scale=1e-3, size=n - 1000)
    north = 10.0 * t
    session = _session(t, gyro, accel, north)

    windows = find_static_window(session, minimum_s=10.0)
    assert windows
    # Detector resolution is 1 s windows, so the boundary is +/- 1 s.
    for start, end in windows:
        assert start >= 1000 - 100  # never deep inside the moving stretch

    mask = quasi_static_mask(session)
    assert mask[:900].mean() < 0.1
    assert mask[1100:].mean() > 0.9

    # Full-session calibration: median biases + leveling + GNSS yaw.
    engine = calibrate_session(session, static_s=10.0)
    assert engine.calibrated
    assert abs(engine.yaw_mount - np.pi / 2.0) < 0.05
    _, accel_v = engine.to_vehicle_frame(session.gyro[1000:], session.accel[1000:])
    np.testing.assert_allclose(np.mean(accel_v, axis=0), [0.0, 0.0, G],
                               atol=0.05)


def test_calibrate_stationary_rejects_short_or_bad_input():
    with pytest.raises(ValueError):
        CalibrationEngine().calibrate_stationary(np.zeros((5, 3)),
                                                 np.zeros((5, 3)))
    with pytest.raises(ValueError):
        CalibrationEngine().calibrate_stationary(np.zeros((50, 3)),
                                                 np.zeros((50, 2)))
