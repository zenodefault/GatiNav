"""Unit tests for the position-domain NHC road update (Phase 2.1)."""

import numpy as np
import pytest

from python.ekf.ekf import ErrorStateEKF


def test_update_nhc_road_pulls_lateral_position_to_road():
    ekf = ErrorStateEKF()
    # start 50 m off the road centreline, road running due east (heading 0)
    ekf.position = np.array([0.0, 50.0, 0.0])
    ekf.update_nhc_road(0.0, lateral_sigma=1.0, confidence=1.0)
    # north component must be pulled toward zero (gain ~0.9 for sigma 1)
    assert abs(ekf.position[1]) < 10.0


def test_update_nhc_road_confidence_scales_correction():
    weak = ErrorStateEKF()
    weak.position = np.array([0.0, 50.0, 0.0])
    weak.update_nhc_road(0.0, lateral_sigma=5.0, confidence=0.1)
    strong = ErrorStateEKF()
    strong.position = np.array([0.0, 50.0, 0.0])
    strong.update_nhc_road(0.0, lateral_sigma=5.0, confidence=1.0)
    # a more confident match must correct further
    assert abs(strong.position[1]) < abs(weak.position[1])


def test_update_nhc_road_heading_orthogonal():
    ekf = ErrorStateEKF()
    # road running north (heading pi/2): the east component is lateral
    ekf.position = np.array([40.0, 0.0, 0.0])
    ekf.update_nhc_road(np.pi / 2.0, lateral_sigma=1.0, confidence=1.0)
    assert abs(ekf.position[0]) < 10.0


def test_update_nhc_road_validation():
    ekf = ErrorStateEKF()
    with pytest.raises(ValueError):
        ekf.update_nhc_road(0.0, lateral_sigma=0.0)
    with pytest.raises(ValueError):
        ekf.update_nhc_road(0.0, lateral_sigma=5.0, confidence=1.5)
    with pytest.raises(ValueError):
        ekf.update_nhc_road(0.0, lateral_sigma=5.0, confidence=0.0)