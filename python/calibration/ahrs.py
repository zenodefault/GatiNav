"""Robust attitude compensation (AHRS) — Blueprint Phase 1.1.

Implements the plan's "Attitude Determination and Compensation" stage:

* ``Ahrs.init_static(accel, mag)`` — from a pre-outage static stretch, the
  accelerometer gravity vector fixes pitch/roll and the magnetometer fixes
  yaw: the level frame is built with the phone's z-axis along gravity (up)
  and its y-axis along magnetic north, exactly as the plan specifies.
* ``Ahrs.update(gyro, accel, mag, dt)`` — a Mahony-style complementary
  filter: the gyroscope propagates the orientation between samples while
  the accelerometer and magnetometer supply corrective feedback.
* Outputs — ``rotation`` (3x3 body->level), ``to_level(accel)`` and
  ``forward_accel(accel)`` (the forward-axis projection the residual net
  consumes in Phase 1.2/1.3).

Conventions: Hamilton quaternions (w, x, y, z) matching
python.ekf.rotations; level frame is ENU with z up and y = magnetic north
at initialisation; ``level = R @ body``.

Magnetometer weight is kept modest (``mag_gain=0.3``): in-vehicle magnetic
fields are corrupted, so the filter leans on gyro+accel during motion and
uses the compass mainly to hold yaw at init and reject slow yaw drift.
"""

import numpy as np

from python.ekf.rotations import (quaternion_to_rotation_matrix,
                                  rotation_matrix_to_quaternion)

KP = 0.5  # proportional correction gain (VERIFY against reference paper)
KI = 0.0  # integral gyro-bias gain (disabled; biases come from calibration)
MAG_GAIN = 0.3  # magnetometer correction weight


def _quat_multiply(a, b):
    """Hamilton quaternion product (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


class Ahrs:
    """Complementary-filter attitude and heading reference system."""

    def __init__(self, kp=KP, ki=KI, mag_gain=MAG_GAIN):
        if kp < 0 or ki < 0 or mag_gain < 0:
            raise ValueError("gains must be non-negative")
        self.kp = float(kp)
        self.ki = float(ki)
        self.mag_gain = float(mag_gain)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.gyro_bias = np.zeros(3)
        self._integral = np.zeros(3)
        self.initialised = False

    # ------------------------------------------------------------------
    # static initialisation (plan: "collect the first 5 seconds of data",
    # gravity -> pitch/roll, magnetometer -> yaw)
    # ------------------------------------------------------------------
    def init_static(self, accel, mag, gyro=None):
        """Initialise attitude from a static stretch of phone-frame data.

        ``accel`` and ``mag`` are (N, 3) arrays (any length >= 5). The mean
        specific force defines the level +z (up); the horizontal component
        of the mean magnetic field defines level +y (magnetic north); x =
        y x z completes the right-handed ENU frame. A per-axis gyro mean
        (pass ``gyro``) is stored as the gyro bias, matching the plan's
        static-period calibration.
        """
        accel = np.asarray(accel, dtype=np.float64)
        mag = np.asarray(mag, dtype=np.float64)
        if accel.ndim != 2 or accel.shape[1] != 3 or accel.shape[0] < 5:
            raise ValueError("accel must have shape (N, 3) with N >= 5")
        if mag.ndim != 2 or mag.shape[1] != 3 or mag.shape[0] < 5:
            raise ValueError("mag must have shape (N, 3) with N >= 5")
        gravity = np.mean(accel, axis=0)
        g_norm = float(np.linalg.norm(gravity))
        if g_norm < 1e-6:
            raise ValueError("static accelerometer magnitude is ~0")
        z = gravity / g_norm
        field = np.mean(mag, axis=0)
        horizontal = field - np.dot(field, z) * z
        h_norm = float(np.linalg.norm(horizontal))
        if h_norm < 1e-6:
            raise ValueError("magnetometer has no horizontal component "
                             "(vertical field or corrupted reading)")
        y = horizontal / h_norm
        x = np.cross(y, z)
        x = x / float(np.linalg.norm(x))
        rotation = np.stack((x, y, z), axis=0)  # rows = level axes in body
        self.q = rotation_matrix_to_quaternion(rotation)
        if gyro is not None:
            gyro = np.asarray(gyro, dtype=np.float64)
            if gyro.shape != accel.shape:
                raise ValueError("gyro must have shape (N, 3)")
            self.gyro_bias = np.mean(gyro, axis=0)
        self._integral = np.zeros(3)
        self.initialised = True
        return self

    # ------------------------------------------------------------------
    # real-time propagation (plan: gyro propagates, accel+mag correct)
    # ------------------------------------------------------------------
    def update(self, gyro, accel, mag=None, dt=0.01):
        """Propagate one sample through the complementary filter.

        Returns the updated quaternion. With ``mag=None`` only the gravity
        direction is corrected (usable when the compass is unavailable).
        """
        if not self.initialised:
            raise RuntimeError("call init_static before update")
        if dt <= 0:
            raise ValueError("dt must be positive")
        gyro = np.asarray(gyro, dtype=np.float64)
        accel = np.asarray(accel, dtype=np.float64)
        if gyro.shape != (3,) or accel.shape != (3,):
            raise ValueError("gyro and accel must be (3,) vectors")

        rotation = quaternion_to_rotation_matrix(self.q)
        # predicted gravity/north directions in the BODY frame
        gravity_body = rotation.T @ np.array([0.0, 0.0, 1.0])
        north_body = rotation.T @ np.array([0.0, 1.0, 0.0])

        error = np.zeros(3)
        a_norm = float(np.linalg.norm(accel))
        if a_norm > 1e-6:
            a_hat = accel / a_norm
            error += np.cross(gravity_body, a_hat)
        if mag is not None:
            m = np.asarray(mag, dtype=np.float64)
            if m.shape != (3,):
                raise ValueError("mag must be a (3,) vector")
            a_ref = a_hat if a_norm > 1e-6 else gravity_body
            horizontal = m - np.dot(m, a_ref) * a_ref
            h_norm = float(np.linalg.norm(horizontal))
            if h_norm > 1e-6:
                error += self.mag_gain * np.cross(north_body,
                                                  horizontal / h_norm)

        self._integral += self.ki * error * dt
        omega = (gyro - self.gyro_bias) + self.kp * error + self._integral
        delta = 0.5 * _quat_multiply(self.q,
                                     np.array([0.0, *omega])) * dt
        q = self.q + delta
        norm = float(np.linalg.norm(q))
        if norm < 1e-12:
            raise RuntimeError("attitude quaternion collapsed")
        self.q = q / norm
        return self.q.copy()

    # ------------------------------------------------------------------
    # outputs
    # ------------------------------------------------------------------
    @property
    def rotation(self):
        """Body->level rotation matrix (3x3)."""
        if not self.initialised:
            raise RuntimeError("call init_static before reading rotation")
        return quaternion_to_rotation_matrix(self.q)

    def to_level(self, accel):
        """Rotate (N, 3) or (3,) accel into the level frame."""
        accel = np.asarray(accel, dtype=np.float64)
        r = self.rotation
        return accel @ r.T if accel.ndim == 2 else r @ accel

    def forward_accel(self, accel):
        """Forward-axis (level y, magnetic-north-aligned) acceleration.

        Gravity lives on level z, so the y component is pure longitudinal
        specific force — exactly the signal the residual net (Phase 1.2)
        denoises and the propagation loop (Phase 1.3) integrates.
        """
        return self.to_level(accel)[..., 1]

    def run(self, gyro, accel, mag=None, dt=0.01):
        """Propagate the whole (N, 3) stream; returns per-sample forward
        acceleration (N,) and the final rotation."""
        gyro = np.asarray(gyro, dtype=np.float64)
        accel = np.asarray(accel, dtype=np.float64)
        if gyro.shape != accel.shape or accel.ndim != 2 or accel.shape[1] != 3:
            raise ValueError("gyro and accel must both have shape (N, 3)")
        out = np.empty(accel.shape[0])
        for i in range(accel.shape[0]):
            m = None if mag is None else mag[i]
            self.update(gyro[i], accel[i], m, dt)
            out[i] = self.forward_accel(accel[i])
        return out