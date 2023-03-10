# This file is for self testing purposes only. Not meant to be used in implementation.

import scipy.linalg as linalg
from scipy.sparse import csr_matrix, bmat
from scipy import signal
import numpy as np


def create_toeplitz(inp, kernel):
    """Creates toeplitz matrix representing 'valid' convolution of matrix with kernel"""
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


def convolve_by_matmult(matrix, inp, kernel_shape):
    inp_shape = inp.shape
    out_height = max(inp_shape[0], kernel_shape[0]) - min(inp_shape[0], kernel_shape[0]) + 1
    out_width = max(inp_shape[1], kernel_shape[1]) - min(inp_shape[1], kernel_shape[1]) + 1

    result_flat = matrix @ inp.flatten()
    result = result_flat.reshape((out_height, out_width))
    return result

def kernel_to_matrix(kernel, mode='Toeplitz'):
    # Get the dimensions of the kernel
    k_rows, k_cols = kernel.shape

    if mode == 'Toeplitz':
        # Create a Toeplitz matrix from the kernel
        first_col = np.concatenate((kernel[:, 0], np.zeros(k_rows - 1)))
        first_row = np.concatenate((kernel[0, :], np.zeros(k_cols - 1)))
        toeplitz_matrix = np.zeros((k_rows * k_cols, k_rows * k_cols))
        for i in range(k_rows):
            for j in range(k_cols):
                index = i * k_cols + j
                row = (j * k_rows) + i
                toeplitz_matrix[index, :] = np.roll(first_row, j * k_rows)[::-1]
                toeplitz_matrix[index, :i] = 0
                toeplitz_matrix[index, i:k_rows] = np.roll(first_col, k_rows - i - 1)[k_rows - i - 1:]
        return toeplitz_matrix
    elif mode == 'Circulant':
        # Create a Circulant matrix from the kernel
        kernel_flip = np.flip(kernel)
        circulant_matrix = np.zeros((k_rows * k_cols, k_rows * k_cols))
        for i in range(k_rows):
            for j in range(k_cols):
                index = i * k_cols + j
                row = np.roll(kernel_flip, i, axis=0)
                col = np.roll(kernel_flip, j, axis=1)
                circulant_matrix[index, :] = row.flatten() @ col.flatten()
        return circulant_matrix
    else:
        raise ValueError('Invalid mode argument. Supported modes: Toeplitz, Circulant')


def convolve(matrix, kernel, mode='valid'):
    return signal.convolve2d(matrix, kernel, mode=mode)


if __name__ == '__main__':
    # I = np.arange(256 * 256).reshape((256, 256))
    I = np.arange(16).reshape(4, 4)
    F = np.array([[10, 20], [30, 40]])
    # F = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
    # F = np.random.randint(0, 100, (4, 4))

    # res = create_toeplitz(I, F)
    res = create_toeplitz(F, I)
    print(res.shape)
    # res_matmult = convolve_by_matmult(res, I, F.shape)
    res_matmult = convolve_by_matmult(res, F, I.shape)
    print(res_matmult)

    res_convolve = convolve(I, np.flip(F))
    print(res_convolve)
