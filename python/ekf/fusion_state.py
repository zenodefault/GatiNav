"""FUSED/INERTIAL/REACQUIRING GNSS<->INS state machine per specs/04 section 4.

Pure state logic; no I/O. The caller drives the machine with strictly
monotonic time (advance), GNSS fix reports (report_fix), and the
covariance-consistency inputs (resolve_consistency). On every state
transition the held 15x15 covariance is scaled by inflation_factor.

Resolved defaults (spec marks these VERIFY; chosen values):
  loss_timeout_s      2.0    spec: "lost for more than 2 s"
  regain_accuracy_m  10.0    spec: regain only when accuracy < 10 m
  accuracy_jump_ratio 3.0    fix accuracy > 3x previous accepted -> jump
  inflation_factor   10.0    scale full covariance on every transition
  consistency_gate    9.0    pass iff NIS = e^T S^-1 e <= gate
"""

from enum import Enum

import numpy as np


class FusionState(Enum):
    FUSED = "FUSED"
    INERTIAL = "INERTIAL"
    REACQUIRING = "REACQUIRING"


class FusionStateMachine:
    def __init__(self, covariance=None, last_fix_t=0.0,
                 last_accepted_accuracy_m=5.0, loss_timeout_s=2.0,
                 regain_accuracy_m=10.0, accuracy_jump_ratio=3.0,
                 inflation_factor=10.0, consistency_gate=9.0):
        if not np.isscalar(last_fix_t) or float(last_fix_t) < 0:
            raise ValueError("last_fix_t must be a non-negative scalar")
        for name, value in (("loss_timeout_s", loss_timeout_s),
                            ("regain_accuracy_m", regain_accuracy_m),
                            ("accuracy_jump_ratio", accuracy_jump_ratio),
                            ("inflation_factor", inflation_factor),
                            ("consistency_gate", consistency_gate)):
            if not np.isscalar(value) or float(value) <= 0:
                raise ValueError(f"{name} must be a positive scalar")
        if not np.isscalar(last_accepted_accuracy_m) or float(last_accepted_accuracy_m) < 0:
            raise ValueError("last_accepted_accuracy_m must be non-negative")
        cov = np.eye(15) if covariance is None else np.asarray(covariance, dtype=np.float64)
        if cov.shape != (15, 15) or not np.all(np.isfinite(cov)):
            raise ValueError("covariance must be a finite (15, 15) array")
        self._cov = cov.copy()
        self._state = FusionState.FUSED
        self._clock = float(last_fix_t)
        self._last_fix_t = float(last_fix_t)
        self._accepted_accuracy = float(last_accepted_accuracy_m)
        self._pending_accuracy = None
        self.loss_timeout_s = float(loss_timeout_s)
        self.regain_accuracy_m = float(regain_accuracy_m)
        self.accuracy_jump_ratio = float(accuracy_jump_ratio)
        self.inflation_factor = float(inflation_factor)
        self.consistency_gate = float(consistency_gate)

    @property
    def state(self):
        return self._state

    @property
    def covariance(self):
        return self._cov.copy()

    def _require_time(self, t):
        if not np.isscalar(t) or not np.isfinite(float(t)):
            raise ValueError("time must be a finite scalar")
        t = float(t)
        if t <= self._clock:
            raise ValueError("time must be strictly increasing")
        self._clock = t
        return t

    def _transition(self, new_state):
        self._cov *= self.inflation_factor
        self._state = new_state

    def advance(self, t):
        """Advance time; FUSED -> INERTIAL once GNSS has been lost > threshold."""
        t = self._require_time(t)
        if (self._state is FusionState.FUSED
                and t - self._last_fix_t > self.loss_timeout_s):
            self._transition(FusionState.INERTIAL)
        return self._state

    def report_fix(self, t, accuracy_m):
        """A GNSS fix arrived at time t with reported accuracy accuracy_m."""
        t = self._require_time(t)
        if not np.isscalar(accuracy_m) or not np.isfinite(float(accuracy_m)):
            raise ValueError("accuracy_m must be a finite scalar")
        accuracy = float(accuracy_m)
        if accuracy < 0:
            raise ValueError("accuracy_m must be non-negative")
        self._last_fix_t = t
        if self._state is FusionState.FUSED:
            if accuracy > self.accuracy_jump_ratio * self._accepted_accuracy:
                # Reported accuracy jumped; treat GNSS as unreliable (§4).
                self._transition(FusionState.INERTIAL)
            else:
                self._accepted_accuracy = accuracy
        elif self._state is FusionState.INERTIAL:
            if accuracy < self.regain_accuracy_m:
                self._pending_accuracy = accuracy
                self._transition(FusionState.REACQUIRING)
        else:  # REACQUIRING: refresh the candidate under check.
            self._pending_accuracy = accuracy
        return self._state

    def resolve_consistency(self, innovation, innovation_covariance):
        """Decide REACQUIRING -> FUSED/INERTIAL from NIS = e^T S^-1 e."""
        if self._state is not FusionState.REACQUIRING:
            raise ValueError("resolve_consistency is only valid in REACQUIRING")
        innovation = np.asarray(innovation, dtype=np.float64)
        S = np.asarray(innovation_covariance, dtype=np.float64)
        if innovation.shape != (3,):
            raise ValueError("innovation must have shape (3,)")
        if S.shape != (3, 3) or not np.allclose(S, S.T):
            raise ValueError("innovation_covariance must be symmetric (3, 3)")
        if not (np.all(np.isfinite(innovation)) and np.all(np.isfinite(S))):
            raise ValueError("innovation and innovation_covariance must be finite")
        try:
            L = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            raise ValueError("innovation_covariance must be positive definite")
        y = np.linalg.solve(L, innovation)
        nis = float(np.dot(y, y))
        if nis <= self.consistency_gate:
            self._accepted_accuracy = self._pending_accuracy
            self._transition(FusionState.FUSED)
        else:
            self._transition(FusionState.INERTIAL)
        self._pending_accuracy = None
        return self._state
