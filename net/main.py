from data import *
from config import Args
from lista import LISTA


if __name__ == "__main__":
    args = Args()

    diag_g = generate_diag_g(args.m, 2)
    A = generate_A(args.m, args.n)

    dataloader = get_lista_dataloader(diag_g, A, args.n, args.p, args.theta, args.batch_size, collate_fn=collate_function)
    model = LISTA(A, diag_g, args.lambd, args.num_layers)

    for batch_idx, (Y, X) in enumerate(dataloader):
        print(Y.shape)
        print(X.shape)
        out = model(Y)
        break
        print(f"{batch_idx} / {len(dataloader)} Y shape: {Y.shape} X shape: {X.shape}")
