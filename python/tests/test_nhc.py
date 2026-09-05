import numpy as np

from python.ekf.ekf import ErrorStateEKF


def test_nhc_reduces_lateral_velocity_and_preserves_along_road_velocity():
    ekf = ErrorStateEKF(
        velocity=[2.0, 3.0, 0.0],
        P0=np.eye(15),
        nhc_sigma=1.0,
    )

    ekf.update_nhc(0.0)

    np.testing.assert_allclose(ekf.velocity, [2.0, 1.5, 0.0])
    assert abs(ekf.velocity[1]) < 3.0


def test_nhc_projects_against_nonzero_road_heading():
    heading = np.pi / 2
    along = np.array([0.0, 4.0, 0.0])
    lateral = np.array([-2.0, 0.0, 0.0])
    ekf = ErrorStateEKF(velocity=along + lateral, P0=np.eye(15))

    ekf.update_nhc(heading)

    np.testing.assert_allclose(ekf.velocity, along + 0.5 * lateral)
