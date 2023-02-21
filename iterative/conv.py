import scipy.linalg as linalg
import numpy as np

def conv_matrix(v, n, mode='valid'):
    """Generates a convolution matrix

    Args:
        v (np.ndarray): vector input
        n (int): length
        mode (str, optional): "full", "same", or "valid". Defaults to 'valid'.

    Returns:
        np.ndarray: A convolution matrix
    """
    X = linalg.convolution_matrix(v, n, mode=mode)
    return X
