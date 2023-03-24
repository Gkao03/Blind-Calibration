import torch
import torch.nn as nn
import numpy as np
import numpy.linalg as la


class LISTA_Layer(nn.Module):
    def __init__(self, B, S, shrink) -> None:
        super(LISTA_Layer, self).__init__()
        self.B = nn.Parameter(B)
        self.S = nn.Parameter(S)
        self.shrink = shrink

    def forward(self, Y):
        C = torch.matmul(self.B, Y) + torch.matmul(self.S, Y)
        X_hat = self.shrink(C)
        return X_hat


class LISTA(nn.Module):
    def __init__(self, A: np.ndarray, diag_g: np.ndarray, lambd, num_layers):
        super(LISTA, self).__init__()
        self.A = A
        self.m, self.n = A.shape
        self.diag_g = diag_g
        self.lambd = lambd
        self.num_layers = num_layers
        self.model = self.build_model()

    def build_model(self):
        B = (self.diag_g @ self.A.T) / (1.01 * la.norm(self.diag_g @ self.A, 2) ** 2)
        S = np.identity(self.n) - np.matmul(B, self.A)
        layers = []

        for _ in range(self.num_layers):
            lista_layer = LISTA_Layer(B, S, nn.Softshrink(self.lambd))
            layers.append(lista_layer)

        return nn.Sequential(*layers)

    def forward(self, Y):
        return self.model(Y)
    

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, X_hat, Y):  # TODO: custom loss
        pass

