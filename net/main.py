from data import *
from utils import get_device, plot_multiple, plot_single
from config import Args
from lista import LISTA
import os
import torch
import torch.nn as nn
import torch.optim as optim


if __name__ == "__main__":
    args = Args()
    np.random.seed(args.random_seed)

    # output dir
    out_dir = os.path.join(args.save_dir, f"exp{args.exp_num}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    diag_g = generate_diag_g(args.m, 2)
    diag_g_init = np.eye(args.m)
    A = generate_A(args.m, args.n)
    train_loader = get_lista_dataloader(diag_g, A, args.n, args.p, args.theta, args.batch_size, collate_fn=collate_function)
    test_loader = get_lista_dataloader(diag_g, A, args.n, 512, args.theta, args.batch_size, collate_fn=collate_function)
    
    device = get_device()
    model = LISTA(A, diag_g_init, args.lambd, args.num_layers).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, verbose=True)
    losses = []

    # for layer_num in range(args.num_layers + 1):

    #     for iter in range(args.epochs_per_layer):
    #         print(f"iteration {iter + 1}/{args.epochs_per_layer}")

    #         for batch_idx, (Y, X) in enumerate(train_loader):
    #             # send to device
    #             Y = Y.to(device)
    #             X = X.to(device)

    #             # zero gradients
    #             optimizer.zero_grad()

    #             # get model output
    #             out, _ = model(Y)

    #             # calculate loss
    #             loss = criterion(out, X)
    #             losses.append(loss.item())

    #             # back prop
    #             loss.backward()
    #             optimizer.step()

    #     scheduler.step()

    #     # freeze layer after training loop
    #     print(f"freezing layer {layer_num}")
    #     for name, param in model.named_parameters():
    #         if param.requires_grad and f'layer{layer_num}' in name:
    #             param.requires_grad = False

    # # save model
    # torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    # # plot losses
    # plot_single(np.arange(len(losses)), losses, "Training Loss", "Iteration", "Loss", os.path.join(out_dir, "loss.png"))


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

    # temp = []
    # for name, param in model.named_parameters():
    #      if 'S' in name:
    #         temp.append(param.data)

    # print(len(temp))
    # print(torch.all(torch.eq(temp[-1], temp[-2])))

    # for name, param in model.named_parameters():
    #     print(name, param.data)
