import torch
from torch.utils.data import DataLoader, Dataset
import scipy.linalg as linalg
import numpy as np


class MyDataset(Dataset):
    def __init__(self, diag_g: np.ndarray, A: np.ndarray, m: np.ndarray, p: np.ndarray, theta: float):
        self.diag_g = diag_g  # calibration
        self.A = A  # measurement matrix
        self.m = m  # dimension of x_i
        self.p = p  # number of measurements
        self.theta = theta  # sparsity level 0 < theta < 1

    def __getitem__(self, index):
        X = generate_X(self.m, self.p, self.theta)
        Y = self.diag_g @ self.A @ X
        return torch.Tensor(Y), torch.Tensor(X)

    def __len__(self):
        return self.p


def generate_X(m, p, theta):
    # generate IID bernoulli-gaussian data x
    X = (0.5 * np.random.randn(m, p) + 0.5 * 1j * np.random.randn(m, p)) * (np.random.rand(m, p) <= theta)
    return X


def generate_diag_g(n, kappa):
    # all_R = np.zeros((n, m))
    # all_normed_R = np.zeros((n, m))

    # F = linalg.dft(m) / np.sqrt(m)
    # FH = F.conj().T

    dft_gain_f = np.random.uniform(0, kappa, n)
    dft_phase_f = np.random.uniform(0, 2 * np.pi, n)
    ground_truth_g = dft_gain_f * np.exp(1j * dft_phase_f)
    diag_g = np.diag(ground_truth_g)

    return diag_g
