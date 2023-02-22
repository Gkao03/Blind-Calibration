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
    print(pad_kernel)

    toeplitz_list = []
    # iterate from last row to first row
    for i in range(pad_kernel.shape[0] - 1, -1, -1):
        c = pad_kernel[i, :] # copy i'th row
        r = np.r_[c[0], np.zeros(in_width - 1)]
        # toeplitz function in scipy
        toeplitz_m = linalg.toeplitz(c, r)
        toeplitz_list.append(toeplitz_m)
        print('F '+ str(i)+'\n', toeplitz_m)

    c = range(1, pad_kernel.shape[0] + 1)
    r = np.r_[c[0], np.zeros(in_width - 1, dtype=int)]
    doubly_indices = linalg.toeplitz(c, r)
    print('doubly indices \n', doubly_indices)

    # shape of one of those toeplitz matrices
    h = toeplitz_list[0].shape[0] * doubly_indices.shape[0]
    w = toeplitz_list[0].shape[1] * doubly_indices.shape[1]
    doubly_blocked_shape = [h, w]
    doubly_blocked = np.zeros(doubly_blocked_shape)

    # tile the toeplitz matrices
    b_h , b_w = toeplitz_list[0].shape # height and width of each block
    for i in range(doubly_indices.shape[0]):
        for j in range(doubly_indices.shape[1]):
            start_i = i * b_h
            start_j = j * b_w
            end_i = start_i + b_h
            end_j = start_j + b_w
            doubly_blocked[start_i:end_i, start_j:end_j] = toeplitz_list[doubly_indices[i, j] - 1]

    print(doubly_blocked @ matrix_to_vector(matrix))
    return doubly_blocked

    # create circulant matrix for each row in kernel
    # circ_mats = [linalg.circulant(row) for row in np.flip(pad_kernel, axis=0)]
    # blocks = []

    # # create block matrix
    # for i in range(len(circ_mats)):
    #     row = circ_mats[:i+1][::-1] + circ_mats[i+1:][::-1]
    #     blocks.append(row)

    # doubly_circulant_mat = np.block(blocks)
    # print(doubly_circulant_mat)

def matrix_to_vector(input):
    input_h , input_w = input.shape
    output_vector = np.zeros(input_h * input_w, dtype=input.dtype)
    # f l i p t h e i n p u t m a t r i x up down
    input = np.flipud(input)
    for i, row in enumerate(input):
        st = i * input_w
        nd = st + input_w
        output_vector[st:nd] = row
    return output_vector


def convolve(matrix, kernel, mode='valid'):
    return signal.convolve2d(matrix, kernel, mode=mode)


if __name__ == '__main__':
    I = np.arange(16).reshape((4, 4))
    F = np.array([[10, 20], [30, 40]])

    create_toeplitz(I, F)
    print(convolve(I, F))
