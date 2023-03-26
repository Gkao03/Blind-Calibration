import torch
import torch.nn as nn
import numpy as np
import numpy.linalg as la


class LossLayer(nn.Module):
    def __init__(self, A, diag_h, theta, norm):
        super(LossLayer, self).__init__()
        self.A = A
        self.diag_h = diag_h
        self.theta = theta
        self.norm = norm

    def forward(self, input):
        m = self.A.shape[0]
        batch_size = input.shape[0]  # batch size
        R = 1 / torch.sqrt((1 / (self.theta * m * batch_size)) * torch.matmul(input, torch.conj(input).mT))
        
        res = torch.matmul(torch.conj(self.A).mT, self.diag_h)
        res = torch.matmul(res, R)
        res = torch.matmul(res, input)

        norm = torch.linalg.norm(torch.real(res), self.norm)  # subtract X from norm to compare?
        self.loss = (1 / batch_size) * torch.pow(norm, 4)
        return input
    

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
    def __init__(self, A: np.ndarray, diag_g: np.ndarray, theta, norm, lambd, num_layers):
        super(LISTA, self).__init__()
        self.A = A
        self.m, self.n = A.shape
        self.diag_g = diag_g
        self.theta = theta
        self.norm = norm
        self.lambd = lambd
        self.num_layers = num_layers

        # build on init
        self.layer_losses = []
        self.model = self.build_model()

    def build_model(self):
        B = (self.diag_g @ self.A.T) / (1.01 * la.norm(self.diag_g @ self.A, 2) ** 2)
        S = np.identity(self.n) - np.matmul(B, self.A)
        layers = []

        for _ in range(self.num_layers):
            lista_layer = LISTA_Layer(B, S, nn.Softshrink(self.lambd))
            layers.append(lista_layer)

            # add loss layer
            loss_layer = LossLayer(self.A, self.diag_g, self.theta, self.norm)
            layers.append(loss_layer)
            self.layer_losses.append(loss_layer)

        return nn.Sequential(*layers)

    def forward(self, Y):
        return self.model(Y)
