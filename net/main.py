from data import *
from lista import LISTA


if __name__ == "__main__":
    m, n, p = 64, 64 * 4, 1024
    diag_g = generate_diag_g(m, 2)
    A = generate_A(m, n)

    dataloader = get_lista_dataloader(diag_g, A, n, p, 0.3, 8, collate_fn=collate_function)

    for batch_idx, (Y, X) in enumerate(dataloader):
        print(f"{batch_idx} / {len(dataloader)}")
