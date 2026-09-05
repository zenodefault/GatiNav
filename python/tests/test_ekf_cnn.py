import numpy as np

from python.ekf.ekf_cnn import CNNEKF


def test_cnn_zupt_wrapper_uses_positive_covariance():
    wrapper = CNNEKF()
    window = np.zeros((100, 6), dtype=np.float64)
    dx = wrapper.update_zupt(window)
    assert dx.shape == (15,)
    assert np.all(np.diag(wrapper.P) >= 0)
