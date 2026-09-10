"""Phase 4B odometry network: causal TCN with heteroscedastic heads.

Input  (B, 6, W): vehicle-frame accel+gyro window (5 s @ 100 Hz).
Output (B, 6):
  0: forward displacement over the window (m), softplus >= 0
  1: log-variance of the displacement head (Gaussian NLL)
  2: forward velocity at window end (m/s), softplus >= 0
  3: log-variance of the velocity head
  4: yaw increment over the window (rad)  (signed, linear)
  5: log-variance of the yaw head

Causality: every convolution is left-padded only, so output at the window
end depends solely on past+present samples -- deployment-shaped for a
sliding 0.5 s stride filter feed.

Param budget: ~430k (well under the 500k AGENTS.md cap), CPU-only ops,
deterministic inference, ONNX/TFLite-exportable (stateless, fixed shapes).
"""

import torch
from torch import nn


class CausalConv1d(nn.Module):
    """Conv1d with left-only padding (causal, no future leakage)."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              dilation=dilation, padding=0)

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    """Residual causal TCN block: Conv-ReLU-Dropout x2 + 1x1 skip."""

    def __init__(self, channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(channels, channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(channels, channels, kernel_size, dilation),
            nn.ReLU(),
        )
        self.skip = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        return self.net(x) + self.skip(x)


class OdomNet(nn.Module):
    """Causal TCN -> forward displacement / velocity / yaw with log-var."""

    CHANNELS = 48
    KERNEL = 7
    DILATIONS = (1, 2, 4, 8, 16)

    def __init__(self, channels=CHANNELS, kernel=KERNEL,
                 dilations=DILATIONS):
        super().__init__()
        self.stem = nn.Sequential(
            CausalConv1d(6, channels, kernel, 1),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[TCNBlock(channels, kernel, d) for d in dilations])
        self.head = nn.Linear(channels, 6)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != 6:
            raise ValueError("x must have shape (batch, 6, W)")
        h = self.blocks(self.stem(x))          # (B, C, W), causal
        features = h[:, :, -1]                  # last-sample state
        out = self.head(features)
        disp = nn.functional.softplus(out[:, 0])
        vel = nn.functional.softplus(out[:, 2])
        return torch.stack((
            disp, out[:, 1], vel, out[:, 3], out[:, 4], out[:, 5],
        ), dim=1)


def gaussian_nll(pred, logvar, target):
    """Mean Gaussian NLL per head (heteroscedastic uncertainty training).

    logvar is clamped to [-6, 6]: unbounded log-variance diverges negative
    on hard batches, exp() underflows and the NLL goes inf -> NaN weights
    (observed: fold B NaN from epoch 1, fold A val spike 3.9 -> 105.9).
    Clamp range spans variances 0.0025..403 -- wide enough for every head.
    """
    logvar = torch.clamp(logvar, min=-6.0, max=6.0)
    return 0.5 * (logvar + (target - pred) ** 2 / logvar.exp()).mean()


PARAM_COUNT = sum(p.numel() for p in OdomNet().parameters())
