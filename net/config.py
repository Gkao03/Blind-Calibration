import os


class Args:
    def __init__(self):
        self.m = 64
        self.n = 64 * 4
        self.p = 2048
        self.theta = 0.3
        self.lambd = 0.1
        self.num_layers = 8
        self.lr = 0.01
        self.batch_size = 16
        self.random_seed = 2023
        self.epochs = 10
        self.epochs_per_layer = 10
        self.save_dir = "out/"
        self.exp_num = 5
        self.out_dir = os.path.join(self.save_dir, f"exp{self.exp_num}")
