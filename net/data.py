import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


def generate_X(m, p, theta):
    # generate IID bernoulli-gaussian data x
    X = (0.5 * np.random.randn(m, p) + 0.5 * 1j * np.random.randn(m, p)) * (np.random.rand(m, p) <= theta)
    return X
