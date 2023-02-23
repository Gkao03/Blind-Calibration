# This file contains some matrix utility functions

import scipy.linalg as linalg
from scipy.sparse import csr_matrix, bmat
from scipy import signal
import numpy as np

def creates_toeplitz1D(v, n, mode='valid'):
    """creates a Toeplitz matrix A such that A @ v is equivalent to using convolve(a, v, mode). 
    The returned array always has n columns. The number of rows depends on the specified mode.
    This function applies for 1D vectors only.    

    Args:
        v (np.ndarray): vector input
        n (int): length
        mode (str, optional): "full", "same", or "valid". Defaults to 'valid'.

    Returns:
        np.ndarray: A convolution matrix. matrix @ v = conv(v, vector of length n)
    """
    X = linalg.convolution_matrix(v, n, mode=mode)
    return X


def create_toeplitz2D(inp, kernel):
    """
    Creates toeplitz matrix representing 2D 'valid' convolution of matrix with kernel.
    """
    in_height, in_width = inp.shape
    kernel_height, kernel_width = kernel.shape

    # if kernel is larger than input
    if kernel_height > in_height or kernel_width > in_width:
        out_height = kernel_height - in_height + 1
        out_width = kernel_width - in_width + 1
        arrs = []

        for i in range(out_height):
            for j in range(out_width):
                arrs.append(kernel[i:i + in_height, j:j + in_width].flatten())

        result_mat = np.vstack(arrs)
        return result_mat

    # input larger than kernel
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
