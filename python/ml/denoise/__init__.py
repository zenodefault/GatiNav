"""Neural residual learning (Blueprint Phase 1.2).

The network learns the *residual error* of the attitude-compensated
forward-axis acceleration — the difference between the measured forward
acceleration and the wheel-encoder-derived true forward acceleration:

    residual = a_measured_forward - a_true_forward

so the propagation loop can compute ``a_denoised = a_measured - residual``
and recover the true signal before double integration (Phase 1.3). This is
a far better-posed target than speed regression: the residual is
near-zero-mean and dominated by structured noise (gravity leakage from
tilt error, mount misalignment, vibration harmonics).
"""