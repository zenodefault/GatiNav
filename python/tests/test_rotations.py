import numpy as np

from python.ekf.rotations import (
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)


def test_known_z_angle_matrices_round_trip():
    values = [
        ([1.0, 0.0, 0.0, 0.0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        ([np.sqrt(3) / 2, 0, 0, 0.5], [[0.5, -0.8660254, 0], [0.8660254, 0.5, 0], [0, 0, 1]]),
        ([np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2], [[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
        ([0.5, 0, 0, np.sqrt(3) / 2], [[-0.5, -0.8660254, 0], [0.8660254, -0.5, 0], [0, 0, 1]]),
        ([0.0, 0.0, 0.0, 1.0], [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
    ]
    for quaternion, expected in values:
        matrix = quaternion_to_rotation_matrix(quaternion)
        assert np.allclose(matrix, expected, atol=1e-7)
        recovered = rotation_matrix_to_quaternion(matrix)
        assert np.allclose(abs(np.dot(recovered, quaternion)), 1.0, atol=1e-7)


def test_ninety_degree_rotations_map_axes():
    half = np.sqrt(2) / 2
    cases = [
        ([half, half, 0, 0], [1, 0, 0], [0, 0, 1]),
        ([half, 0, half, 0], [0, 0, -1], [0, 1, 0]),
        ([half, 0, 0, half], [0, 1, 0], [-1, 0, 0]),
    ]
    for quaternion, x_expected, y_expected in cases:
        matrix = quaternion_to_rotation_matrix(quaternion)
        assert np.allclose(matrix @ [1, 0, 0], x_expected)
        assert np.allclose(matrix @ [0, 1, 0], y_expected)
