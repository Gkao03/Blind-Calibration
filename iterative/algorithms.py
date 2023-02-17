import scipy.linalg as linalg
import numpy as np


def u_step(g: np.ndarray, H: np.ndarray, Dx: np.ndarray, Dy: np.ndarray, alpha: float, gamma: float, n: int) -> np.ndarray:
    vx = np.zeros(n)
    vy = np.zeros(n)
    ax = np.zeros(n)
    ay = np.zeros(n)
    
    left = H.T @ H + (alpha / gamma) * (Dx.T @ Dx + Dy.T @ Dy)
    right = H.T @ g + (alpha / gamma) * (Dx.T @ (vx + ax) + Dy.T @ (vy + ay))
    u = linalg.inv(left) @ right
