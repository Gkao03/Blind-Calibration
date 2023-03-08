import cv2
import numpy as np
import scipy
from utils import *


if __name__ == '__main__':
    im_arr = read_image("../orig_images/lenna.png", flag=cv2.IMREAD_GRAYSCALE)
    np_to_image_save(im_arr, "test.png")
