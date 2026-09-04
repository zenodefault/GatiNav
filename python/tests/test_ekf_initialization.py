import numpy as np
import pytest

from python.ekf.ekf import ErrorStateEKF


def test_initializes_the_15_element_state_and_covariance():
    state = np.arange(15, dtype=np.float64)
    covariance = np.eye(15)
    ekf = ErrorStateEKF(state, covariance)
    assert np.array_equal(ekf.error_state, state)
    assert np.array_equal(ekf.P, covariance)
    assert np.array_equal(ekf.position_error, state[6:9])


@pytest.mark.parametrize(
    "state, covariance",
    [(np.zeros(14), np.eye(15)), (np.zeros(15), np.eye(14))],
)
def test_rejects_invalid_shapes(state, covariance):
    with pytest.raises(ValueError):
        ErrorStateEKF(state, covariance)
