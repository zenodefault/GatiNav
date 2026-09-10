"""Denoised EKF propagation (Blueprint Phase 1.3).

Wraps the 15-state ErrorStateEKF so its propagation uses the denoised
forward-axis acceleration exactly as the plan's Phase 1.3 workflow:

  a. raw accelerometer reading
  b. AHRS attitude compensation -> level frame
  c. project onto the forward axis -> forward_accel_raw
  d. residual net -> residual_pred (run at a stride, held between calls)
  e. forward_accel_denoised = forward_accel_raw - residual_pred
  f. the corrected accel is rotated back to the body frame and fed to the
     underlying EKF's predict(), which integrates it (twice, through its
     velocity and position states).

The AHRS is initialised from a pre-outage static stretch
(``init_static(accel, mag)``), mirroring the plan's startup sequence.
"""

import numpy as np
import torch

from python.calibration.ahrs import Ahrs
from python.ekf.ekf import ErrorStateEKF
from python.ml.denoise.model import ResidualTCN

FS = 100.0
MODEL_STRIDE_S = 0.5  # residual net inference stride (held between calls)


class DenoisedEKF:
    """ErrorStateEKF whose propagation consumes denoised forward accel."""

    def __init__(self, model=None, mean=None, std=None,
                 ahrs=None, model_stride_s=MODEL_STRIDE_S, fs=FS,
                 **ekf_kwargs):
        self.ekf = ErrorStateEKF(**ekf_kwargs)
        self.ahrs = Ahrs() if ahrs is None else ahrs
        self.model = ResidualTCN() if model is None else model
        self.model.eval()
        self.mean = 0.0 if mean is None else float(mean)
        self.std = 1.0 if std is None else float(std)
        if self.std <= 0:
            raise ValueError("std must be positive")
        self.fs = float(fs)
        self.model_stride = max(1, int(round(model_stride_s * self.fs)))
        self._buffer = np.zeros(self.model_stride * 5 + 1)
        self._n = 0
        self._residual = 0.0
        self.initialised = False

    def init_static(self, accel, mag, gyro=None):
        """Initialise AHRS from a pre-outage static stretch (plan Phase 1.1)."""
        self.ahrs.init_static(accel, mag, gyro=gyro)
        self.initialised = True
        return self

    def predict(self, gyro, accel, mag=None, dt=None):
        """Propagate one sample with the denoised forward acceleration."""
        if not self.initialised:
            raise RuntimeError("call init_static before predict")
        if dt is None:
            dt = 1.0 / self.fs
        # b) attitude compensation -> level frame
        self.ahrs.update(np.asarray(gyro), np.asarray(accel), mag, dt)
        level = self.ahrs.to_level(accel)
        # c) forward-axis projection (level y)
        forward_raw = level[1]
        # d) residual at the model stride, held between calls
        self._buffer = np.roll(self._buffer, -1)
        self._buffer[-1] = forward_raw
        self._n += 1
        if self._n % self.model_stride == 0:
            with torch.no_grad():
                x = torch.from_numpy(
                    ((self._buffer.astype(np.float32) - self.mean)
                     / self.std).reshape(1, 1, -1))
                self._residual = float(self.model(x)[0, 0])
        # e) denoise the forward component; f) rotate back and integrate
        level_denoised = level.copy()
        level_denoised[1] -= self._residual
        accel_corrected = self.ahrs.rotation.T @ level_denoised
        return self.ekf.predict(gyro, accel_corrected, dt)

    def __getattr__(self, name):
        return getattr(self.ekf, name)