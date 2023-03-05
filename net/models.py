import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self, x):
        pass


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, x):
        pass


class LISTA(nn.Module):
    def __init__(self, input_size, dict_size, num_layers):
        super(LISTA, self).__init__()
        self.input_size = input_size
        self.dict_size = dict_size
        self.num_layers = num_layers
        
        # Define the weight matrices and bias vectors
        self.W = nn.Parameter(torch.randn(dict_size, input_size))
        self.b = nn.Parameter(torch.randn(dict_size, 1))
        self.Theta = nn.Parameter(torch.randn(num_layers, dict_size, dict_size))
        self.Psi = nn.Parameter(torch.randn(num_layers, dict_size, dict_size))
        self.alpha = nn.Parameter(torch.randn(num_layers, dict_size, 1))
        
    def forward(self, x):
        # Flatten the input tensor
        x = x.view(-1, self.input_size, 1)
        
        # Linear transform using the dictionary
        z = F.linear(x, self.W, self.b)
        
        # Iterative calculation of sparse codes
        for i in range(self.num_layers):
            z = F.linear(z, self.Theta[i], None) + F.linear(x, self.alpha[i], None)
            z = F.softshrink(z, lambd=0.1)
            z = F.linear(z, self.Psi[i], None)
            z = F.relu(z)
            
        # Return the final sparse code
        return z.view(-1, self.dict_size)


class AdaptiveLISTA(nn.Module):
    def __init__(self, input_size, dict_size, num_layers):
        super(AdaptiveLISTA, self).__init__()
        
        self.input_size = input_size
        self.dict_size = dict_size
        self.num_layers = num_layers
        
        # Initialize the weight matrix W and bias vector b
        self.W = nn.Parameter(torch.Tensor(dict_size, input_size))
        self.b = nn.Parameter(torch.Tensor(dict_size, 1))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        nn.init.zeros_(self.b)
        
        # Initialize the thresholding function T
        self.T = nn.Threshold(0, 0)
        
        # Initialize the ALISTA layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(dict_size, dict_size))
        
    def forward(self, x):
        # Flatten the input tensor
        x = x.view(-1, self.input_size)
        
        # Compute the sparse code using the ALISTA algorithm
        z = torch.zeros(x.shape[0], self.dict_size).to(x.device)
        z_prev = z
        alpha = 1.0 / torch.norm(self.W, p=2)
        for i in range(self.num_layers):
            z = self.layers[i](z_prev)
            z = z + torch.mm(x - torch.mm(z, self.W.t()), self.W) * alpha
            z = self.T(z)
            z_prev = z
        
        # Reconstruct the input from the sparse code and dictionary
        x_recon = torch.mm(z, self.W) + self.b.t()
        x_recon = x_recon.view(-1, 1, int(math.sqrt(self.input_size)), int(math.sqrt(self.input_size)))
        
        return x_recon


class ALISTALayer(nn.Module):
    def __init__(self, dict_size):
        super(ALISTALayer, self).__init__()
        self.dict_size = dict_size
        self.W = nn.Parameter(torch.randn(self.dict_size, self.dict_size))
        self.b = nn.Parameter(torch.randn(self.dict_size))
        self.T = nn.Softshrink()

    def forward(self, z):
        z = torch.matmul(z, self.W) + self.b
        z = self.T(z)
        return z


class ALISTANet(nn.Module):
    def __init__(self, dict_size, num_layers, input_size):
        super(ALISTANet, self).__init__()
        self.dict_size = dict_size
        self.num_layers = num_layers
        self.input_size = input_size
        
        # Initialize the dictionary and biases
        self.W = nn.Parameter(torch.randn(self.input_size, self.dict_size))
        self.A = nn.Parameter(torch.eye(self.dict_size) - torch.randn(self.dict_size, self.dict_size) * 0.1)
        self.c = nn.Parameter(torch.randn(self.dict_size))
        self.b = nn.Parameter(torch.randn(self.input_size))
        
        # Initialize the thresholding function
        self.T = nn.Softshrink()
        
        # Compute the analytical weights
        I = torch.eye(self.dict_size)
        WtW = torch.matmul(self.W.t(), self.W)
        S = torch.inverse(I + torch.matmul(self.A.t(), self.A) + torch.matmul(WtW, self.A.t()))
        T = torch.matmul(WtW, self.A.t())
        self.W_a = nn.Parameter(torch.matmul(T, S))
        self.A_a = nn.Parameter(torch.matmul(self.A, S))
        self.c_a = nn.Parameter(torch.matmul(self.c, S))
        
        # Initialize the layers of the network
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(ALISTALayer(self.dict_size))
    
    def forward(self, x):
        # Flatten the input tensor
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        
        # Compute the sparse code using the ALISTA algorithm with analytical weights
        z = torch.zeros(batch_size, self.dict_size).to(x.device)
        z_prev = z
        for i in range(self.num_layers):
            z = self.layers[i](z_prev)
            z = z + torch.matmul(x - torch.matmul(z, self.W_a.t()) - torch.matmul(z_prev - z, self.A_a), self.W_a) + torch.matmul(z_prev - z, self.c_a)
            z = self.T(z)
            z_prev = z
        
        # Reconstruct the input from the sparse code and dictionary
        x_recon = torch.matmul(z, self.W) + self.b
        x_recon = x_recon.view(batch_size, 1, int(math.sqrt(self.input_size)), int(math.sqrt(self.input_size)))
        
        return x_recon
