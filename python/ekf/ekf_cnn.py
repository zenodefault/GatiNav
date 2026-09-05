"""CNN-driven ZUPT covariance wrapper around the baseline ENU EKF."""

import numpy as np
import torch

from python.ml.cnn import NoiseNet
from python.ekf.ekf import ErrorStateEKF


class CNNEKF:
    def __init__(self, model=None, mean=None, std=None, **ekf_kwargs):
        self.ekf = ErrorStateEKF(**ekf_kwargs)
        self.model = NoiseNet() if model is None else model
        self.model.eval()
        self.mean = np.zeros(6) if mean is None else np.asarray(mean, dtype=np.float64)
        self.std = np.ones(6) if std is None else np.asarray(std, dtype=np.float64)
        if self.mean.shape != (6,) or self.std.shape != (6,) or np.any(self.std <= 0):
            raise ValueError("mean and std must have shape (6,) and positive std")

    def predict(self, gyro, accel, dt):
        return self.ekf.predict(gyro, accel, dt)

    def update_zupt(self, imu_window):
        window = np.asarray(imu_window, dtype=np.float64)
        if window.shape != (100, 6):
            raise ValueError("imu_window must have shape (100, 6) in Acc-Gyro order")
        values = (window - self.mean) / self.std
        with torch.no_grad():
            covariance = self.model(
                torch.from_numpy(values.T[None]).to(dtype=torch.float32)
            ).cpu().numpy()[0]
        return self.ekf.update_zupt(covariance)

    def __getattr__(self, name):
        return getattr(self.ekf, name)
