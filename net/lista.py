import torch
import torch.nn as nn


class LISTA_Layer(nn.Module):
    def __init__(self, B, S, shrink) -> None:
        super(LISTA_Layer, self).__init__()
        self.B = nn.Parameter(B)
        self.S = nn.Parameter(S)
        self.shrink = shrink

    def forward(self, X):
        C = self.B + torch.matmul(self.S, X)
        X_hat = self.shrink(C)
        return X_hat


class LISTA(nn.Module):
    def __init__(self, A, diag_g, lambd, num_layers):
        super(LISTA, self).__init__()
        self.A = A
        self.m, self.n = A.shape
        self.diag_g = diag_g
        self.lambd = lambd
        self.num_layers = num_layers
        self.model = self.build_model()

    def build_model(self):
        pass

    def forward(self, Y):
        pass
