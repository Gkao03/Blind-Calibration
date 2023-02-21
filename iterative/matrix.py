# This file contains some matrix utility functions

import scipy.linalg as linalg
import numpy as np

def conv_matrix(v, n, mode='valid'):
    """Generates a convolution matrix

    Args:
        v (np.ndarray): vector input
        n (int): length
        mode (str, optional): "full", "same", or "valid". Defaults to 'valid'.

    Returns:
        np.ndarray: A convolution matrix. matrix @ v = conv(v, vector of length n)
    """
    X = linalg.convolution_matrix(v, n, mode=mode)
    return X
