import numpy as np


def mse(X: np.array, y: np.array, theta: np.array) -> np.float64:
    """Compute cost by Andrew Ng's modified mean squared error.

    Args:
        X: training examples, with bias term added.
        y: labeled output for training examples
        theta: parameters

    Returns:
        MSE for given x, y, and theta
    """
    m = y.shape[0]
    scale_factor = 1 / (2 * m)
    predictions = X.dot(theta).sum(axis=1)
    errors = predictions - y
    squared_errors = errors ** 2
    sum_squared_errors = squared_errors.sum()
    return scale_factor * sum_squared_errors


def gradient_descent(
    X: np.array, y: np.array, alpha: float, theta: np.array, num_iters: int,
) -> np.array:
    """Perform gradient descent.

    Args:
        X: training examples, with bias term added.
        y: labeled output for training examples
        alpha: learning rate
        theta: parameters
        num_iters: number of iterations of gradient descent to perform.

    Returns:
        New parameters optimized over num_iters of gradient descent.
    """
    m = y.shape[0]
    scale_factor = alpha / m
    for i in range(num_iters):
        hypothesis = X.dot(theta).reshape(-1)
        error = hypothesis - y
        new_t0 = scale_factor * (error * X[:, 0]).sum()
        new_t1 = scale_factor * (error * X[:, 1]).sum()
        theta[0] = theta[0] - new_t0
        theta[1] = theta[1] - new_t1
    return theta
