import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def np_to_image_save(np_array, path):
    im = Image.fromarray(np_array)
    im.save(path)
