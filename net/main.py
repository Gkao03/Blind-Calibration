from data import *
from utils import get_device, plot_multiple, plot_single, plot_hist
from config import Args
from lista import LISTA
import os
import torch
import torch.nn as nn
import torch.optim as optim


def train_freeze(args, model, train_loader, optimizer, scheduler, criterion, device):
    model = model.to(device)
    model.train()

    for layer_num in range(args.num_layers + 1):

        for iter in range(args.epochs_per_layer):
            print(f"iteration {iter + 1}/{args.epochs_per_layer}")

            for batch_idx, (Y, X) in enumerate(train_loader):
                # send to device
                Y = Y.to(device)
                X = X.to(device)

                # zero gradients
                optimizer.zero_grad()

                # get model output
                out, _ = model(Y)

                # calculate loss
                loss = criterion(out, X)
                losses.append(loss.item())

                # back prop
                loss.backward()
                optimizer.step()

        scheduler.step()

        # freeze layer after training loop
        print(f"freezing layer {layer_num}")
        for name, param in model.named_parameters():
            if param.requires_grad and f'layer{layer_num}' in name:
                param.requires_grad = False

    # save model
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    # plot losses
    plot_single(np.arange(len(losses)), losses, "Training Loss", "Iteration", "Loss", os.path.join(out_dir, "loss.png"))


if __name__ == "__main__":
    args = Args()
    np.random.seed(args.random_seed)

    # output dir
    out_dir = os.path.join(args.save_dir, f"exp{args.exp_num}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    diag_g = generate_diag_g(args.m, np.random.uniform(0, 50))
    diag_g_init = generate_diag_g(args.m, np.random.uniform(0, 50))
    A = generate_A(args.m, args.n)
    train_loader = get_lista_dataloader(diag_g, A, args.n, args.p, args.theta, args.batch_size, collate_fn=collate_function)
    test_loader = get_lista_dataloader(diag_g, A, args.n, 512, args.theta, 1, collate_fn=collate_function)
    
    device = get_device()
    model = LISTA(A, diag_g_init, args.lambd, args.num_layers)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, verbose=True)
    losses = []

    # training with intermediate recon layers
    for epoch in range(args.epochs):
        print(f"epoch {epoch + 1}/{args.epochs}")

        for batch_idx, (Y, X) in enumerate(train_loader):
            # send to device
            Y = Y.to(device)
            X = X.to(device)

            # zero gradients
            optimizer.zero_grad()

            # get model output
            out, _ = model(Y)

            # get recons
            recon_layers = model.get_recons()
            loss = 0

            # calculate loss
            for recon_layer in recon_layers:
                loss += criterion(recon_layer.get_recon(), X)

            losses.append(loss.item() / args.num_layers)

            # back prop
            loss.backward()
            optimizer.step()

        scheduler.step()

    # save model
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    # plot losses
    plot_single(np.arange(len(losses)), losses, "Training Loss", "Iteration", "Loss", os.path.join(out_dir, "loss.png"))

    # evaluation
    model.eval()
    recon_losses = []

    for batch_idx, (Y, X) in enumerate(test_loader):
        Y = Y.to(device)
        X = X.to(device)

        out, _ = model(Y)

        recon_loss = criterion(out, X)
        recon_losses.append(recon_loss.item())

    # plot recon losses
    plot_hist(recon_losses, title="Test Data Reconstruction L1 Error", xlabel="L1 Error", ylabel="Density", savefile=os.path.join(out_dir, "test_recon_loss.png"))
    
    # temp = []
    # for name, param in model.named_parameters():
    #      if 'S' in name:
    #         temp.append(param.data)

    # print(len(temp))
    # print(torch.all(torch.eq(temp[-1], temp[-2])))

    # for name, param in model.named_parameters():
    #     print(name, param.data)
