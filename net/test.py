import cv2
import numpy as np
import scipy
from utils import *


if __name__ == '__main__':
    im_arr = read_image("../orig_images/lenna.png", flag=cv2.IMREAD_GRAYSCALE)
    np_to_image_save(im_arr, "test.png")  # original

    im_fft = np.fft.fft2(im_arr)
    im_fft_shift = np.fft.fftshift(im_fft)
    np_to_image_save(np.log(np.abs(im_fft_shift)), "test_fft.png")
    # np_to_image_save(np.log(np.abs(im_fft)), "test_fft2.png")

    # im_phase_shift = translate_by_phase_shift(im_arr, 10*np.pi, 10*np.pi)
    # np_to_image_save(np.log(np.abs(im_phase_shift)), "test_fft.png")

    im_phase_shift = phase_shift(im_arr, 100, 100)
    np_to_image_save(np.log(np.abs(im_phase_shift)), "test_shift.png")

    im_fft = np.fft.fftn(im_phase_shift)
    im_fft_shift = np.fft.fftshift(im_fft)
    np_to_image_save(np.log(np.abs(im_fft_shift)), "test_shift_fft.png")

    # misc unused inference for result
    # m = self.A.shape[0]
    # batch_size = input.shape[0]  # batch size
    # R = 1 / torch.sqrt((1 / (self.theta * m * batch_size)) * torch.matmul(input, torch.conj(input).mT))
    
    # res = torch.matmul(torch.conj(self.A).mT, self.diag_h)
    # res = torch.matmul(res, R)
    # res = torch.matmul(res, input)

