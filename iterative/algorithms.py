import scipy.linalg as linalg
import numpy as np


class U_Step:
    def __init__(self, g: np.ndarray, H: np.ndarray, Dx: np.ndarray, Dy: np.ndarray, alpha: float, gamma: float, n: int):
        self.vx = np.zeros(n)
        self.vy = np.zeros(n)
        self.ax = np.zeros(n)
        self.ay = np.zeros(n)
        self.g = g
        self.H = H
        self.Dx = Dx
        self.Dy = Dy
        self.alpha = alpha
        self.gamma = gamma

        # calculations
        self.left = H.T @ H + (alpha / gamma) * (Dx.T @ Dx + Dy.T @ Dy)
        self.HT_g = H.T @ g

        # calc v vectorize function
        self.vfunc = np.vectorize(calc_v)

    def step(self) -> np.ndarray:        
        right = self.HT_g + (self.alpha / self.gamma) * (self.Dx.T @ (self.vx + self.ax) + self.Dy.T @ (self.vy + self.ay))
        u = linalg.inv(self.left) @ right


def calc_v(s, arr):
    pass
