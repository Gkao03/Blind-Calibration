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

    # flip ud
    input = np.flipud(input)
    for i, row in enumerate(input):
        st = i * input_w
        nd = st + input_w
        output_vector[st:nd] = row
    return output_vector


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
    I = np.arange(16).reshape((4, 4))
    F = np.array([[10, 20], [30, 40]])

    # create_toeplitz(I, F)
    kernel = np.array([[1, 2], [3, 4]])
    input_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    toeplitz_matrix = kernel_to_matrix(kernel, mode='Toeplitz')
    circulant_matrix = kernel_to_matrix(kernel, mode='Circulant')
    input_flat = input_matrix.flatten()
    output_conv = np.convolve(input_flat, kernel.flatten(), mode='valid')
    output_toeplitz = toeplitz_matrix @ input_flat
    output_circulant = circulant_matrix @ input_flat
    print(output_conv.reshape((2, 2)))
    print(output_toeplitz.reshape((2, 2)))
    print(output_circulant.reshape((2, 2)))
    print(convolve(I, F))
