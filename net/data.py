import torch
import numpy as np


def generate_X(m, p, theta):
    # generate IID bernoulli-gaussian data x
    X = (0.5 * np.random.randn(m, p) + 0.5 * 1j * np.random.randn(m, p)) * (np.random.rand(m, p) <= theta)
    return X
