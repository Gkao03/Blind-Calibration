import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import imageio
from PIL import Image


def np_to_image_save(np_array, path):
    imageio.imwrite(path, np_array)


def read_image(path, flag):
    im = cv2.imread(path, flag)
    return im
