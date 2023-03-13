import torch
import torch.nn as nn


class LISTA_Layer(nn.Module):
    def __init__(self, B, S, shrink) -> None:
        super(LISTA_Layer, self).__init__()
        self.B = B
        self.S = S
        self.shrink = shrink

    def forward(self, X):
        pass