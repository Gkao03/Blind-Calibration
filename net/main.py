from data import *
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
    model = LISTA(A, diag_g, args.lambd, args.num_layers)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.epochs):
        for batch_idx, (Y, X) in enumerate(dataloader):
            optimizer.zero_grad()

            # get model output
            out, _ = model(Y)

            # get recons
            recon_layers = model.get_recons()
            loss = 0

            # calculate loss
            for recon_layer in recon_layers:
                loss += criterion(recon_layer.get_recon(), X)

            # back prop
            loss.backward()
            optimizer.step()
