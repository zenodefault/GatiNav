"""Unit tests for FusionStateMachine.report_health (Blueprint Phase 2.2)."""

import numpy as np
import pytest

from python.ekf.fusion_state import FusionState, FusionStateMachine


def test_unhealthy_fix_drops_fused_to_inertial():
    machine = FusionStateMachine(last_fix_t=0.0)
    machine.advance(1.0)
    assert machine.state is FusionState.FUSED
    # degraded PDOP while "fused" must drop to INERTIAL immediately
    state = machine.report_health(2.0, pdop=12.0, satellites=3, snr=18.0)
    assert state is FusionState.INERTIAL


def test_healthy_fix_in_inertial_enters_reacquiring():
    machine = FusionStateMachine(last_fix_t=0.0)
    machine.advance(5.0)  # lost > 2 s -> INERTIAL
    assert machine.state is FusionState.INERTIAL
    state = machine.report_health(6.0, pdop=1.5, satellites=9, snr=40.0,
                                  accuracy_m=3.0)
    assert state is FusionState.REACQUIRING


def test_health_resolution_back_to_fused():
    machine = FusionStateMachine(last_fix_t=0.0)
    machine.advance(5.0)
    machine.report_health(6.0, pdop=1.5, satellites=9, snr=40.0,
                          accuracy_m=3.0)
    state = machine.resolve_consistency(
        np.zeros(3), np.eye(3) * 1.0)  # NIS = 0 <= gate
    assert state is FusionState.FUSED


def test_health_with_partial_metrics_skips_missing():
    machine = FusionStateMachine(last_fix_t=0.0)
    machine.advance(5.0)  # INERTIAL
    # only accuracy provided -> considered healthy (no thresholds to fail)
    state = machine.report_health(6.0, accuracy_m=4.0)
    assert state is FusionState.REACQUIRING


def test_reacquiring_inflates_covariance():
    machine = FusionStateMachine(last_fix_t=0.0)
    cov_before = machine.covariance
    machine.advance(5.0)  # INERTIAL transition inflates
    machine.report_health(6.0, pdop=1.5, satellites=9, snr=40.0,
                          accuracy_m=3.0)
    cov_after = machine.covariance
    assert np.all(cov_after >= cov_before)


def test_report_health_requires_increasing_time():
    machine = FusionStateMachine(last_fix_t=5.0)
    with pytest.raises(ValueError):
        machine.report_health(5.0, pdop=2.0)