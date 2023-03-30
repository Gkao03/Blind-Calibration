import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import imageio
import matplotlib.pyplot as plt
from PIL import Image


def plot_hist(x, title="title", xlabel="x label", ylabel="y label", savefile="plot.png", ylim=None):
    plt.style.use('tableau-colorblind10')

    plt.hist(x, density=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if ylim is not None:
        plt.ylim(ylim)
    
    plt.savefig(savefile)
    print("saved plot")
    plt.close()


def plot_multiple(x, ys, legends, title="title", xlabel="x label", ylabel="y label", savefile="plot.png", ylim=None):
    plt.style.use('tableau-colorblind10')
    
    for y, legend in zip(ys, legends):
        plt.plot(x, y, linewidth=1, label=legend)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best', fontsize='small')

    if ylim is not None:
        plt.ylim(ylim)
    
    plt.savefig(savefile)
    print("saved plot")
    plt.close()


def plot_single(x, y, title="title", xlabel="x label", ylabel="y label", savefile="plot.png", ylim=None):
    plt.style.use('tableau-colorblind10')

    plt.plot(x, y, linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if ylim is not None:
        plt.ylim(ylim)
    
    plt.savefig(savefile)
    print("saved plot")
    plt.close()


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def np_to_image_save(np_array, path):
    imageio.imwrite(path, np_array)


def read_image(path, flag):
    im = cv2.imread(path, flag)
    return im


def phase_shift(fimage, dx, dy):
    # Shift the phase of the fourier transform of an image
    dims = fimage.shape
    x, y = np.meshgrid(np.arange(-dims[1] / 2, dims[1] / 2), np.arange(-dims[0] / 2, dims[0] / 2))

    kx = -1j * 2 * np.pi * x / dims[1]
    ky = -1j * 2 * np.pi * y / dims[0]

    shifted_fimage = fimage * np.exp(-(kx * dx + ky * dy))

    return shifted_fimage


def translate_by_phase_shift(image, dx, dy):
    # Get the fourier transform
    fimage = np.fft.fftshift(np.fft.fftn(image))
    # Phase shift
    shifted_fimage = phase_shift(fimage, dx, dy)
    # Inverse transform -> translated image
    shifted_image = np.fft.ifftn(np.fft.ifftshift(shifted_fimage))

    return shifted_image
