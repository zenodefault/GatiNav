"""Unit tests for python/matching/scenario.py (Blueprint plan 3.3)."""

import numpy as np
import pytest

from python.matching.scenario import detect_scenario


def _signal(duration_s=120, fs=10):
    n = int(duration_s * fs)
    return np.arange(n) / fs, np.zeros(n), np.ones(n, dtype=bool)


def test_highway_detection():
    t, speed, valid = _signal()
    speed[30 * 10:90 * 10] = 25.0  # 60 s of sustained highway speed
    labels, structured = detect_scenario(t, speed, valid)
    assert (labels == "highway").sum() >= 20
    assert np.all(structured == (labels != "urban"))


def test_tunnel_detection():
    t, speed, valid = _signal()
    speed[:] = 20.0  # moving throughout
    valid[20 * 10:50 * 10] = False  # 30 s GNSS blackout at speed
    labels, structured = detect_scenario(t, speed, valid)
    tunnel = np.flatnonzero(labels == "tunnel")
    assert tunnel.size >= 10
    # the blackout interior must be labelled tunnel
    assert labels[25 * 10] == "tunnel"


def test_stopped_blackout_is_not_tunnel():
    t, speed, valid = _signal()
    speed[:] = 0.0  # parked (e.g. in a garage) -> not a tunnel
    valid[20 * 10:50 * 10] = False
    labels, _ = detect_scenario(t, speed, valid)
    assert not np.any(labels == "tunnel")


def test_urban_default():
    t, speed, valid = _signal()
    speed[:] = 8.0
    labels, structured = detect_scenario(t, speed, valid)
    assert np.all(labels == "urban")
    assert not np.any(structured)


def test_empty_and_shape_validation():
    with pytest.raises(ValueError):
        detect_scenario(np.zeros(3), np.zeros(2), np.ones(3, dtype=bool))
    labels, structured = detect_scenario(
        np.array([]), np.array([]), np.array([], dtype=bool))
    assert labels.size == 0 and structured.size == 0