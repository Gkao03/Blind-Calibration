import torch
import torch.nn as nn


class LISTA_Layer(nn.Module):
    def __init__(self, Y, X, We, S, theta) -> None:
        super(LISTA_Layer, self).__init__()
        self.Y = Y
        self.X = X
        self.We = We
        self.S = S
        self.theta = theta

    def forward(self, x):
        pass