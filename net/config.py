class Args:
    def __init__(self):
        self.m = 64
        self.n = 64 * 4
        self.p = 1024
        self.theta = 0.3
        self.lambd = 0.1
        self.num_layers = 4
        self.lr = 0.001
        self.batch_size = 8
        self.random_seed = 2023
        self.epochs = 10
        self.iters_per_layer = 5
        self.save_dir = "./out"
        self.exp_num = 1
