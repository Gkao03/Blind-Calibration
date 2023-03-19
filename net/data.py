import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np


class MyDataset(Dataset):
    def __init__(self, diag_g, A, m, p, theta):
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
