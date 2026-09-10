"""Phase 4B learned odometry (python/ml/odom/).

Components: vehicle-frame window dataset with wheel-derived targets
(dataset), causal-TCN odometry network with heteroscedastic heads (model),
driver-held-out training (train), and the open-loop gate evaluation
(eval_odom). Supervision uses IO-VNBD wheel-encoder-derived velocity /
GNSS-free GT only during TRAINING; inference inputs are IMU-only.
"""
