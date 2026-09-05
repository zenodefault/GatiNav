"""NoiseNet architecture for positive ZUPT measurement variances."""

import torch
from torch import nn


class NoiseNet(nn.Module):
    """1D CNN mapping six-channel IMU windows to three variances."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1600, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softplus(),
        )

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (6, 100):
            raise ValueError("x must have shape (batch, 6, 100)")
        return self.head(self.features(x))
