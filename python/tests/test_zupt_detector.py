import numpy as np

from python.eval.engine import (_stationarity_scores, _stationary_mask,
                                ACCEL_VAR_MAX, GYRO_VAR_MAX)


def test_constant_noisy_signal_has_low_temporal_scores():
    rng = np.random.default_rng(4)
    accel = np.tile([0.0, 0.0, 9.80665], (300, 1))
    gyro = np.zeros((300, 3))
    accel += rng.normal(0.0, 0.01, accel.shape)
    gyro += rng.normal(0.0, 0.0001, gyro.shape)
    av, gv = _stationarity_scores(accel, gyro)
    assert np.median(av[99:]) < ACCEL_VAR_MAX
    assert np.median(gv[99:]) < GYRO_VAR_MAX


def test_oscillating_signal_has_orders_larger_scores():
    t = np.arange(300, dtype=np.float64) / 100.0
    accel = np.tile([0.0, 0.0, 9.80665], (300, 1))
    accel[:, 0] += 0.5 * np.sin(2.0 * np.pi * 2.0 * t)
    gyro = np.zeros((300, 3))
    av, gv = _stationarity_scores(accel, gyro)
    np.testing.assert_allclose(np.median(av[99:]), 0.125, atol=1e-12)


def test_detector_length_and_warmup_alignment():
    accel = np.tile([0.0, 0.0, 9.80665], (150, 1))
    gyro = np.zeros((150, 3))
    mask = _stationary_mask(accel, gyro)
    assert mask.shape == (150,)
    assert not np.any(mask[:99])
    assert np.all(mask[99:])
