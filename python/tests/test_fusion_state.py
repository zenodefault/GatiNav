"""Tests for specs/04 section 4: FUSED/INERTIAL/REACQUIRING state machine."""

import numpy as np
import pytest

from python.ekf.fusion_state import FusionState, FusionStateMachine

I = np.eye(15)


def _machine(**kwargs):
    kwargs.setdefault("last_accepted_accuracy_m", 5.0)
    return FusionStateMachine(**kwargs)


def test_starts_fused():
    m = _machine()
    assert m.state is FusionState.FUSED
    assert np.allclose(m.covariance, I)


def test_fused_to_inertial_after_loss_over_2s():
    m = _machine()
    assert m.report_fix(1.0, 5.0) is FusionState.FUSED
    assert m.advance(3.0) is FusionState.FUSED  # gap == 2.0 s: not > 2 s
    assert m.advance(3.0001) is FusionState.INERTIAL  # gap > 2 s


def test_fused_to_inertial_on_accuracy_jump():
    m = _machine()
    assert m.report_fix(1.0, 5.0) is FusionState.FUSED
    before = m.covariance
    # 16.0 > 3 x 5.0 even though a fix is present.
    assert m.report_fix(2.0, 16.0) is FusionState.INERTIAL
    assert np.allclose(m.covariance, before * m.inflation_factor)


def test_gnss_flapping_stays_fused():
    m = _machine()
    for t in np.arange(0.5, 3.0, 0.5):
        assert m.report_fix(float(t), 5.0) is FusionState.FUSED
    assert np.allclose(m.covariance, I)  # no transition -> no inflation
    # A real (> 2 s) outage finally drops to INERTIAL.
    assert m.advance(5.5) is FusionState.INERTIAL


def test_regain_with_bad_accuracy_rejected():
    m = _machine()
    m.report_fix(1.0, 5.0)
    assert m.advance(4.0) is FusionState.INERTIAL
    before = m.covariance
    # Accuracy >= 10 m is not a valid regain; rejected without transition.
    assert m.report_fix(5.0, 25.0) is FusionState.INERTIAL
    assert np.allclose(m.covariance, before)
    # Boundary accuracy == 10.0 m is also rejected (spec: "< 10 m").
    assert m.report_fix(6.0, 10.0) is FusionState.INERTIAL
    assert np.allclose(m.covariance, before)


def test_inertial_to_reacquiring_on_good_fix():
    m = _machine()
    m.report_fix(1.0, 5.0)
    assert m.advance(4.0) is FusionState.INERTIAL
    before = m.covariance
    assert m.report_fix(5.0, 8.0) is FusionState.REACQUIRING
    assert np.allclose(m.covariance, before * m.inflation_factor)


def test_reacquiring_to_fused_when_consistent():
    m = _machine()
    m.report_fix(1.0, 5.0)
    m.advance(4.0)
    m.report_fix(5.0, 8.0)
    before = m.covariance
    # NIS = e^T S^-1 e with e = (3,0,0), S = I equals 9.0 <= gate 9.0: pass.
    assert m.resolve_consistency([3.0, 0.0, 0.0], np.eye(3)) is FusionState.FUSED
    assert np.allclose(m.covariance, before * m.inflation_factor)
    # The reacquired fix is now the accepted reference for jump checks.
    assert m.report_fix(6.0, 25.0) is FusionState.INERTIAL  # > 3 x 8.0


def test_reacquiring_to_inertial_when_inconsistent():
    m = _machine()
    m.report_fix(1.0, 5.0)
    m.advance(4.0)
    m.report_fix(5.0, 8.0)
    before = m.covariance
    # NIS = 16.0 > gate 9.0: rejected, back to INERTIAL.
    assert m.resolve_consistency([4.0, 0.0, 0.0], np.eye(3)) is FusionState.INERTIAL
    assert np.allclose(m.covariance, before * m.inflation_factor)
    # A later good fix may try reacquisition again.
    assert m.report_fix(6.0, 6.0) is FusionState.REACQUIRING
    assert m.resolve_consistency([0.0, 0.0, 0.0], np.eye(3)) is FusionState.FUSED


def test_covariance_inflated_on_every_transition():
    m = _machine()
    expect = 1.0
    m.report_fix(1.0, 5.0)          # FUSED stays
    m.advance(4.0)                  # FUSED -> INERTIAL
    expect *= m.inflation_factor
    assert np.allclose(m.covariance, I * expect)
    m.report_fix(5.0, 8.0)          # INERTIAL -> REACQUIRING
    expect *= m.inflation_factor
    assert np.allclose(m.covariance, I * expect)
    m.resolve_consistency([0.0, 0.0, 0.0], np.eye(3))  # REACQUIRING -> FUSED
    expect *= m.inflation_factor
    assert np.allclose(m.covariance, I * expect)
    m.advance(8.0)                  # FUSED -> INERTIAL (fix at t=5, gap 3 s)
    assert m.state is FusionState.INERTIAL
    expect *= m.inflation_factor
    assert np.allclose(m.covariance, I * expect)


def test_consistency_only_in_reacquiring():
    m = _machine()
    with pytest.raises(ValueError):
        m.resolve_consistency([0.0, 0.0, 0.0], np.eye(3))
    m.report_fix(1.0, 5.0)
    m.advance(4.0)  # INERTIAL
    with pytest.raises(ValueError):
        m.resolve_consistency([0.0, 0.0, 0.0], np.eye(3))


def test_time_must_be_strictly_monotonic():
    m = _machine()
    m.report_fix(1.0, 5.0)
    with pytest.raises(ValueError):
        m.advance(0.5)  # going backward
    with pytest.raises(ValueError):
        m.report_fix(1.0, 5.0)  # same timestamp again


def test_validation():
    with pytest.raises(ValueError):
        _machine(covariance=np.eye(3))
    with pytest.raises(ValueError):
        _machine(loss_timeout_s=0.0)
    with pytest.raises(ValueError):
        _machine(inflation_factor=-1.0)
    with pytest.raises(ValueError):
        _machine(last_accepted_accuracy_m=-1.0)
    m = _machine()
    with pytest.raises(ValueError):
        m.report_fix(1.0, -2.0)
    m.report_fix(2.0, 5.0)
    m.advance(5.0)
    m.report_fix(6.0, 8.0)
    with pytest.raises(ValueError):
        m.resolve_consistency([0.0, 0.0, 0.0], np.eye(2))  # wrong shape
    with pytest.raises(ValueError):
        m.resolve_consistency(np.zeros(3), np.array([[1.0, 2.0, 0.0],
                                                     [0.0, 1.0, 0.0],
                                                     [0.0, 0.0, 1.0]]))
    with pytest.raises(ValueError):
        m.resolve_consistency(np.zeros(3), np.array([[1.0, 2.0, 0.0],
                                                     [2.0, 1.0, 0.0],
                                                     [0.0, 0.0, 1.0]]))  # not PD
