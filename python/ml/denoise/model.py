"""Residual-learning network (Blueprint Phase 1.2): causal TCN.

Input  (B, 1, W): attitude-compensated forward-axis acceleration window.
Output (B, 1):    scalar residual correction (linear head; residuals are
                  signed), which the propagation loop subtracts from the
                  measured forward accel before integration.

Causality: every convolution is left-padded only, so the output at the
window end depends solely on past+present samples — deployment-shaped for a
sliding 0.5 s stride feed. The plan suggests a TCN or a shallow MLP; the
TCN is chosen for temporal context with a tiny parameter budget (~70k,
well under any on-device cap) and deterministic ONNX-exportable inference.
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


class ResidualTCN(nn.Module):
    """Causal TCN: forward-accel window -> scalar residual correction."""

    CHANNELS = 24
    KERNEL = 7
    DILATIONS = (1, 2, 4, 8)

    def __init__(self, channels=CHANNELS, kernel=KERNEL,
                 dilations=DILATIONS):
        super().__init__()
        self.stem = nn.Sequential(
            CausalConv1d(1, channels, kernel, 1),
            nn.ReLU(),
        )
        blocks = []
        for d in dilations:
            blocks += [
                CausalConv1d(channels, channels, kernel, d),
                nn.ReLU(),
                CausalConv1d(channels, channels, kernel, d),
                nn.ReLU(),
            ]
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, 1)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("x must have shape (batch, 1, W)")
        h = self.blocks(self.stem(x))
        return self.head(h[:, :, -1])  # (B, 1), causal


PARAM_COUNT = sum(p.numel() for p in ResidualTCN().parameters())