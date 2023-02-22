import scipy.linalg as linalg
from scipy import signal
import numpy as np


def create_toeplitz(matrix, kernel):
    """Creates toeplitz matrix representing 'valid' convolution of matrix with kernel"""
    in_height, in_width = matrix.shape
    kernel_height, kernel_width = kernel.shape

    out_height = in_height - kernel_height + 1
    out_width = in_width - kernel_width + 1

    # zero pad kernel (to size of 'valid' output)
    pad_kernel = np.pad(kernel, ((max(out_height - kernel_height, 0), 0), (0, max(out_width - kernel_width, 0))), 'constant')
    # print(pad_kernel)

    # create toeplitz matrix for each row of kernel
    # toeplitz_mats = [linalg.toeplitz(row) for row in pad_kernel]
    # for row in toeplitz_mats:
    #     print(row)

    # create circulant matrix for each row in kernel
    circ_mats = [linalg.circulant(row) for row in np.flip(pad_kernel, axis=0)]
    blocks = []

    # create block matrix
    for i in range(len(circ_mats)):
        row = circ_mats[:i+1][::-1] + circ_mats[i+1:][::-1]
        blocks.append(row)

    doubly_circulant_mat = np.block(blocks)
    print(doubly_circulant_mat)


def convolve(matrix, kernel, mode='valid'):
    return signal.convolve2d(matrix, kernel, mode=mode)


if __name__ == '__main__':
    I = np.arange(16).reshape((4, 4))
    F = np.array([[10, 20], [30, 40]])

    print(create_toeplitz(I, F))
    print(convolve(I, F))
