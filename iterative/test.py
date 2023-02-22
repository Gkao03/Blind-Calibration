# This file is for self testing purposes only. Not meant to be used in implementation.

import scipy.linalg as linalg
from scipy.sparse import csr_matrix
from scipy import signal
import numpy as np


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
        toep_mat = linalg.toeplitz(c, row)
        toeplitz_matrices.append(toep_mat)

    # create new matrix from toeplitz matrices
    num_block_rows = out_size // toeplitz_matrices[0].shape[0]
    blocks = []
    i = len(toeplitz_matrices)

    for _ in range(num_block_rows):
        block_row = toeplitz_matrices[i:] + toeplitz_matrices[:i]
        blocks.append(block_row)
        i -= 1

    result_mat = np.block(blocks)
    return result_mat


def convolve_by_matmult(matrix, inp, kernel_shape):
    inp_shape = inp.shape
    out_height = inp_shape[0] - kernel_shape[0] + 1
    out_width = inp_shape[1] - kernel_shape[1] + 1

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
    I = np.arange(256 * 256).reshape((256, 256))
    # F = np.array([[10, 20], [30, 40]])
    F = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
    # F = np.random.randint(0, 100, (4, 4))

    res = create_toeplitz(I, F)
    print(res.shape)
    res_matmult = convolve_by_matmult(res, I, F.shape)
    print(res_matmult)

    res_convolve = convolve(I, np.flip(F))
    print(res_convolve)

    # kernel = np.array([[1, 2], [3, 4]])
    # input_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # toeplitz_matrix = kernel_to_matrix(kernel, mode='Toeplitz')
    # circulant_matrix = kernel_to_matrix(kernel, mode='Circulant')
    # input_flat = input_matrix.flatten()
    # output_conv = np.convolve(input_flat, kernel.flatten(), mode='valid')
    # output_toeplitz = toeplitz_matrix @ input_flat
    # output_circulant = circulant_matrix @ input_flat
    # print(output_conv.reshape((2, 2)))
    # print(output_toeplitz.reshape((2, 2)))
    # print(output_circulant.reshape((2, 2)))
    # print(convolve(I, F))
