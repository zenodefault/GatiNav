"""NoiseNet architecture: IMU windows -> ZUPT variances + forward speed."""

import torch
from torch import nn


class NoiseNet(nn.Module):
    """1D CNN mapping six-channel IMU windows to three variances + speed.

    Output columns 0..2 are per-axis ZUPT measurement variances (Softplus,
    positive). Column 3 is a forward-speed head regressing log(1 + v), so it
    is Softplus-shaped and non-negative; speed = expm1(output) at inference.
    Both heads share the convolutional trunk and the 1600->64 projection.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1600, 64),
            nn.ReLU(),
        )
        self.var_head = nn.Sequential(
            nn.Linear(64, 3),
            nn.Softplus(),
        )
        self.speed_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (6, 100):
            raise ValueError("x must have shape (batch, 6, 100)")
        h = self.trunk(self.features(x))
        return torch.cat((self.var_head(h), self.speed_head(h)), dim=1)
