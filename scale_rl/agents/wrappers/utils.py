"""
This implementation is adapted from the Gymnasium API.
ref: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/wrappers/utils.py
"""

import numpy as np


class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=(), dtype=np.float32):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, dtype=dtype)
        self.var = np.ones(shape, dtype=dtype)
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count
    ratio = batch_count / tot_count

    new_mean = mean + delta * ratio
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * ratio
    new_var = M2 / tot_count

    return new_mean, new_var, tot_count
