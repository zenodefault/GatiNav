"""15-state ENU error-state EKF using the supplied project equations."""

import numpy as np

from .rotations import quaternion_to_rotation_matrix


class ErrorStateEKF:
    STATE_SIZE = 15
    GRAVITY = np.array([0.0, 0.0, -9.80665])

    def __init__(self, initial_error_state=None, P0=None, rotation=None,
                 velocity=None, position=None, gyro_bias=None,
                 accel_bias=None, noise_variances=None, nhc_sigma=1.0):
        self.error_state = np.zeros(15) if initial_error_state is None else self._vec(
            initial_error_state, (15,), "initial_error_state")
        self.P = np.eye(15) if P0 is None else self._mat(P0, (15, 15), "P0")
        self.rotation = np.eye(3) if rotation is None else self._mat(
            rotation, (3, 3), "rotation")
        self.velocity = self._vec(np.zeros(3) if velocity is None else velocity, (3,), "velocity")
        self.position = self._vec(np.zeros(3) if position is None else position, (3,), "position")
        self.gyro_bias = self._vec(np.zeros(3) if gyro_bias is None else gyro_bias, (3,), "gyro_bias")
        self.accel_bias = self._vec(np.zeros(3) if accel_bias is None else accel_bias, (3,), "accel_bias")
        self.noise_variances = np.zeros(4) if noise_variances is None else self._vec(
            noise_variances, (4,), "noise_variances")
        if not np.isscalar(nhc_sigma) or nhc_sigma <= 0:
            raise ValueError("nhc_sigma must be a positive scalar")
        self.nhc_sigma = float(nhc_sigma)
        self.orientation = self.rotation
        self.orientation_error = self.error_state[0:3]
        self.velocity_error = self.error_state[3:6]
        self.position_error = self.error_state[6:9]
        self.gyro_bias_error = self.error_state[9:12]
        self.accel_bias_error = self.error_state[12:15]

    @staticmethod
    def _vec(value, shape, name):
        result = np.asarray(value, dtype=np.float64)
        if result.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        return result.copy()

    @staticmethod
    def _mat(value, shape, name):
        result = np.asarray(value, dtype=np.float64)
        if result.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        return result.copy()

    @staticmethod
    def _skew(vector):
        x, y, z = vector
        return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)

    @staticmethod
    def _exp_rotation(vector):
        angle = np.linalg.norm(vector)
        if angle < 1e-12:
            return np.eye(3) + ErrorStateEKF._skew(vector)
        axis = vector / angle
        cross = ErrorStateEKF._skew(axis)
        return np.cos(angle) * np.eye(3) + (1 - np.cos(angle)) * np.outer(axis, axis) + np.sin(angle) * cross

    def predict(self, gyro, accel, dt):
        gyro = self._vec(gyro, (3,), "gyro")
        accel = self._vec(accel, (3,), "accel")
        if not np.isscalar(dt) or dt <= 0:
            raise ValueError("dt must be a positive scalar")
        omega = gyro - self.gyro_bias
        specific_force = accel - self.accel_bias
        previous_rotation = self.rotation
        self.rotation = previous_rotation @ self._exp_rotation(omega * dt)
        self.orientation = self.rotation
        midpoint_rotation = previous_rotation @ self._exp_rotation(omega * (0.5 * dt))
        acceleration = midpoint_rotation @ specific_force + self.GRAVITY
        self.position = self.position + self.velocity * dt + 0.5 * acceleration * dt * dt
        self.velocity = self.velocity + acceleration * dt
        phi = np.eye(15)
        phi[0:3, 0:3] -= self._skew(omega) * dt
        phi[0:3, 9:12] = -np.eye(3) * dt
        phi[3:6, 0:3] = -self.rotation @ self._skew(specific_force) * dt
        phi[3:6, 12:15] = -self.rotation * dt
        phi[6:9, 3:6] = np.eye(3) * dt
        self.P = phi @ self.P @ phi.T + self._process_noise(dt)
        self.P = (self.P + self.P.T) / 2
        return self.position.copy()

    def _process_noise(self, dt):
        q = np.zeros((15, 15))
        blocks = (0, 3, 9, 12)
        for value, start in zip(self.noise_variances, blocks):
            q[start:start + 3, start:start + 3] = np.eye(3) * value * dt
        return q

    def update(self, measurement, predicted, H, R):
        z = self._vec(measurement, np.asarray(measurement).shape, "measurement")
        h = self._vec(predicted, z.shape, "predicted")
        H = np.asarray(H, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)
        if H.ndim != 2 or H.shape[1] != 15 or H.shape[0] != z.size:
            raise ValueError("H must have shape (measurement_size, 15)")
        if R.shape != (z.size, z.size):
            raise ValueError("R must be square with measurement size")
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        dx = K @ (z - h)
        self.rotation = self._exp_rotation(dx[0:3]) @ self.rotation
        self.velocity += dx[3:6]
        self.position += dx[6:9]
        self.gyro_bias += dx[9:12]
        self.accel_bias += dx[12:15]
        identity = np.eye(15)
        residual = identity - K @ H
        self.P = residual @ self.P @ residual.T + K @ R @ K.T
        self.P = (self.P + self.P.T) / 2
        return dx

    def update_gnss(self, position_enu, accuracy):
        position = self._vec(position_enu, (3,), "position_enu")
        if not np.isscalar(accuracy) or accuracy <= 0:
            raise ValueError("accuracy must be a positive scalar")
        H = np.zeros((3, 15))
        H[:, 6:9] = np.eye(3)
        R = np.eye(3) * float(accuracy) ** 2
        return self.update(position, self.position, H, R)

    def inject_zero_velocity(self, covariance):
        covariance = self._vec(covariance, (3,), "covariance")
        if np.any(covariance <= 0):
            raise ValueError("covariance entries must be positive")
        H = np.zeros((3, 15))
        H[:, 3:6] = np.eye(3)
        return self.update(np.zeros(3), self.velocity, H, np.diag(covariance))

    def update_zupt(self, covariance=(1.0, 1.0, 1.0)):
        """Apply the supplied zero-velocity pseudo-measurement covariance."""
        return self.inject_zero_velocity(covariance)

    def update_nhc(self, road_heading):
        """Pull ENU velocity toward the matched road's along-heading axis."""
        if not np.isscalar(road_heading):
            raise ValueError("road_heading must be a scalar")
        heading = float(road_heading)
        lateral = np.array([-np.sin(heading), np.cos(heading), 0.0])
        H = np.zeros((1, 15))
        H[0, 3:6] = lateral
        predicted = np.array([lateral @ self.velocity])
        return self.update(
            np.zeros(1),
            predicted,
            H,
            np.array([[self.nhc_sigma ** 2]]),
        )
