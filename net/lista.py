import torch
import torch.nn as nn


class LISTA_Layer(nn.Module):
    def __init__(self, A, B, theta, shrink) -> None:
        super(LISTA_Layer, self).__init__()
        self.A = A
        self.B = B
        self.theta = theta
        self.shrink = shrink

    def forward(self, x):
        pass