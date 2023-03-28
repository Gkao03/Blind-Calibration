from data import *
from utils import get_device
from config import Args
from lista import LISTA
import torch
import torch.nn as nn
import torch.optim as optim


if __name__ == "__main__":
    args = Args()
    np.random.seed(args.random_seed)

    diag_g = generate_diag_g(args.m, 2)
    A = generate_A(args.m, args.n)
    dataloader = get_lista_dataloader(diag_g, A, args.n, args.p, args.theta, args.batch_size, collate_fn=collate_function)
    
    device = get_device()
    model = LISTA(A, diag_g, args.lambd, args.num_layers).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for layer_num in range(args.num_layers + 1):

        for iter in range(args.iters_per_layer):
            print(f"iteration {iter + 1}/{args.iters_per_layer}")

            for batch_idx, (Y, X) in enumerate(dataloader):
                # send to device
                Y = Y.to(device)
                X = X.to(device)

                # zero gradients
                optimizer.zero_grad()

                # get model output
                out, _ = model(Y)

                # calculate loss
                loss = criterion(out, X)

                # back prop
                loss.backward()
                optimizer.step()

        # freeze layer after training loop
        print(f"freezing layer {layer_num}")
        for name, param in model.named_parameters():
            if param.requires_grad and f'layer{layer_num}' in name:
                param.requires_grad = False

    # training with intermediate recon layers
    # for epoch in range(args.epochs):
    #     for batch_idx, (Y, X) in enumerate(dataloader):
    #         optimizer.zero_grad()

    #         # get model output
    #         out, _ = model(Y)

    #         # get recons
    #         recon_layers = model.get_recons()
    #         loss = 0

    #         # calculate loss
    #         for recon_layer in recon_layers:
    #             loss += criterion(recon_layer.get_recon(), X)

    #         # back prop
    #         loss.backward()
    #         optimizer.step()

    # temp = []
    # for name, param in model.named_parameters():
    #      if 'S' in name:
    #         temp.append(param.data)

    # print(len(temp))
    # print(torch.all(torch.eq(temp[-1], temp[-2])))

    # for name, param in model.named_parameters():
    #     print(name, param.data)
