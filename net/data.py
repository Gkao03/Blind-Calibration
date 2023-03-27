import torch
from torch.utils.data import DataLoader, Dataset
import scipy.linalg as linalg
import numpy as np


class MyDataset(Dataset):
    def __init__(self, diag_g: np.ndarray, A: np.ndarray, n: np.ndarray, p: np.ndarray, theta: float):
        self.diag_g = diag_g  # calibration
        self.A = A  # measurement matrix
        self.n = n  # dimension of x_i
        self.p = p  # number of measurements
        self.theta = theta  # sparsity level 0 < theta < 1

    def __getitem__(self, index):
        X = generate_X(self.n, 1, self.theta)
        Y = self.diag_g @ self.A @ X
        return torch.Tensor(Y), torch.Tensor(X)

    def __len__(self):
        return self.p


def collate_function(batch):
    Y, X = zip(*batch)
    Y_stack = torch.stack(Y, dim=0)
    X_stack = torch.stack(X, dim=0)
    return Y_stack, X_stack


def get_lista_dataloader(diag_g: np.ndarray, A: np.ndarray, m: np.ndarray, p: np.ndarray, theta: float, batch_size: int, collate_fn):
    my_dataset = MyDataset(diag_g, A, m, p, theta)
    dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    return dataloader


def generate_X(n, p, theta):
    # generate IID bernoulli-gaussian data x
    X = (0.5 * np.random.randn(n, p) + 0.5 * 1j * np.random.randn(n, p)) * (np.random.rand(n, p) <= theta)
    return X


def generate_A(m, n):
    assert m <= n, "m should be less than equal to n"

    dft_mat = linalg.dft(n) / np.sqrt(n)
    random_select_row = np.random.permutation(n)
    A = dft_mat[random_select_row[:m], :]

    return A


def generate_diag_g(m, kappa):
    dft_gain_f = np.random.uniform(0, kappa, m)
    dft_phase_f = np.random.uniform(0, 2 * np.pi, m)
    ground_truth_g = dft_gain_f * np.exp(1j * dft_phase_f)
    diag_g = np.diag(ground_truth_g)

    return diag_g
