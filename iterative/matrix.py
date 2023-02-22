# This file contains some matrix utility functions

import scipy.linalg as linalg
from scipy.sparse import csr_matrix, bmat
from scipy import signal
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


def create_toeplitz(inp, kernel):
    """Creates toeplitz matrix representing 'valid' convolution of matrix with kernel"""
    in_height, in_width = inp.shape
    kernel_height, kernel_width = kernel.shape

    out_height = in_height - kernel_height + 1
    out_width = in_width - kernel_width + 1
    out_size = out_height * out_width

    # zero pad kernel
    pad_kernel = np.pad(kernel, [(0, in_height - kernel_height), (0, out_width - 1)])

    # create toeplitz matrix for each row of padded kernel
    toeplitz_matrices = []

    for row in pad_kernel:
        c = np.r_[row[0], [0] * (out_width - 1)]
        toep_mat = csr_matrix(linalg.toeplitz(c, row))
        toeplitz_matrices.append(toep_mat)

    # create new matrix from toeplitz matrices
    num_block_rows = out_size // toeplitz_matrices[0].shape[0]
    blocks = []
    i = len(toeplitz_matrices)

    for _ in range(num_block_rows):
        block_row = toeplitz_matrices[i:] + toeplitz_matrices[:i]
        blocks.append(block_row)
        i -= 1

    result_mat = bmat(blocks)
    return result_mat