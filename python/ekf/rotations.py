"""Hamilton quaternion utilities for body-to-world ENU rotations."""

import numpy as np


def quaternion_to_rotation_matrix(quaternion):
    """Return R(q) for a Hamilton quaternion ordered (w, x, y, z)."""
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError("quaternion must have shape (4,)")
    norm = np.linalg.norm(q)
    if norm == 0.0:
        raise ValueError("quaternion must be nonzero")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_quaternion(rotation):
    """Return a Hamilton quaternion ordered (w, x, y, z)."""
    r = np.asarray(rotation, dtype=np.float64)
    if r.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    if not np.allclose(r @ r.T, np.eye(3), atol=1e-10):
        raise ValueError("rotation must be orthonormal")
    if not np.isclose(np.linalg.det(r), 1.0, atol=1e-10):
        raise ValueError("rotation must have determinant 1")
    trace = np.trace(r)
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = np.array(
            [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
        )
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = ((1, 2) if i == 0 else (2, 0) if i == 1 else (0, 1))
        s = 2.0 * np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k])
        q = np.zeros(4)
        q[i + 1] = 0.25 * s
        q[0] = (r[k, j] - r[j, k]) / s
        q[j + 1] = (r[j, i] + r[i, j]) / s
        q[k + 1] = (r[k, i] + r[i, k]) / s
    q /= np.linalg.norm(q)
    return q if q[0] >= 0.0 else -q
