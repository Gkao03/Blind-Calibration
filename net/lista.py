import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.linalg as la


class ReconLayer(nn.Module):
    def __init__(self):
        super(ReconLayer, self).__init__()

    def get_recon(self):
        return self.recon

    def forward(self, input):
        self.recon = input[0]
        return input
    

class LISTA_Layer1(nn.Module):
    def __init__(self, B, shrink):
        super(LISTA_Layer1, self).__init__()
        self.B = nn.Parameter(B)
        self.shrink = shrink

    def forward(self, Y):
        By = torch.matmul(self.B, Y)
        By_real = By.real
        By_imag = By.imag

        X_hat_real = self.shrink(By_real)
        X_hat_imag = self.shrink(By_imag)
        X_hat = torch.complex(X_hat_real, X_hat_imag)

        return X_hat, By
    

class LISTA_Layer(nn.Module):
    def __init__(self, S, shrink):
        super(LISTA_Layer, self).__init__()
        self.S = nn.Parameter(S)
        self.shrink = shrink

    def forward(self, input):
        X_in, By = input

        C = torch.matmul(self.S, X_in) + By
        C_real = C.real
        C_imag = C.imag

        X_hat_real = self.shrink(C_real)
        X_hat_imag = self.shrink(C_imag)
        X_hat = torch.complex(X_hat_real, X_hat_imag)

        return X_hat, By


# TODO: check model definition for using diag_g
class LISTA(nn.Module):
    def __init__(self, A: np.ndarray, diag_g: np.ndarray, lambd, num_layers):
        super(LISTA, self).__init__()
        self.A = A
        self.m, self.n = A.shape
        self.diag_g = diag_g
        self.lambd = lambd
        self.num_layers = num_layers

        # build on init
        self.recon_layers = []  # may or may not be used depending on version
        self.model = nn.Sequential()
        self.build_model_v2()

    def build_model_v1(self):
        B = (self.diag_g @ self.A).T / (1.01 * la.norm(self.diag_g @ self.A, 2) ** 2)
        S = np.identity(self.n) - np.matmul(B, self.A)

        # convert to tensors
        B = torch.tensor(B)
        S = torch.tensor(S)

        # initial layers
        self.model.add_module(LISTA_Layer1(B.detach().clone(), nn.Softshrink(self.lambd)))
        recon_layer = ReconLayer()
        self.model.add_module('recon_layer0', recon_layer)
        self.recon_layers.append(recon_layer)

        for i in range(self.num_layers):
            lista_layer = LISTA_Layer(S.detach().clone(), nn.Softshrink(self.lambd))
            self.model.add_module(f'layer{i + 1}', lista_layer)

            # add recon layer
            recon_layer = ReconLayer()
            self.model.add_module(f'recon_layer{i + 1}', recon_layer)
            self.recon_layers.append(recon_layer)
    
    def build_model_v2(self):
        B = (self.diag_g @ self.A).T / (1.01 * la.norm(self.diag_g @ self.A, 2) ** 2)
        S = np.identity(self.n) - np.matmul(B, self.A)

        # convert to tensors
        B = torch.tensor(B)
        S = torch.tensor(S)

        self.model.add_module('layer0', LISTA_Layer1(B.detach().clone(), nn.Softshrink(self.lambd)))

        for i in range(self.num_layers):
            lista_layer = LISTA_Layer(S.detach().clone(), nn.Softshrink(self.lambd))
            self.model.add_module(f'layer{i + 1}', lista_layer)
    
    def get_recons(self):
        return self.recon_layers

    def forward(self, Y):
        return self.model(Y)
