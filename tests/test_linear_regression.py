import numpy as np

from models.linear_regression import mse


def test_mse_not_negative():
    """Test that the MSE function is not negative."""
    X = np.random.random((2, 2))
    y = np.random.random((2, 1))
    theta = np.random.random((2, 1))
    assert mse(X=X, y=y, theta=theta) >= 0
