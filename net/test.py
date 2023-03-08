import cv2
import numpy as np
import scipy
from utils import *


if __name__ == '__main__':
    im_arr = read_image("../orig_images/lenna.png", flag=cv2.IMREAD_GRAYSCALE)
    # np_to_image_save(im_arr, "test.png")

    im_fft = np.fft.fft2(im_arr)
    im_fft_shift = np.fft.fftshift(im_fft)
    # np_to_image_save(np.log(np.abs(im_fft_shift)), "test_fft.png")
    # np_to_image_save(np.log(np.abs(im_fft)), "test_fft2.png")
