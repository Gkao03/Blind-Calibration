import scipy.linalg as linalg
from scipy import signal
import numpy as np


def create_toeplitz(matrix, kernel):
    """Creates toeplitz matrix representing 'valid' convolution of matrix with kernel"""
    in_height, in_width = matrix.shape
    kernel_height, kernel_width = kernel.shape

    out_height = in_height - kernel_height + 1
    out_width = in_width - kernel_width + 1

    print(kernel.shape)
    print(out_height, out_width)

    pad_kernel = np.pad(kernel, ((max(out_height - kernel_height, 0), 0), (0, max(out_width - kernel_width, 0))), 'constant')
    return pad_kernel

def convolve(matrix, kernel, mode='valid'):
    return signal.convolve2d(matrix, kernel, mode=mode)


if __name__ == '__main__':
    I = np.arange(16).reshape((4, 4))
    F = np.array([[10, 20], [30, 40]])

    print(create_toeplitz(I, F))
    print(convolve(I, F))
