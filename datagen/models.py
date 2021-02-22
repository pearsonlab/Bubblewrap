import numpy as np

def lorenz(t, y: np.ndarray, s=10, r=28, b=2.667):
    x_dot = s * (y[1] - y[0])
    y_dot = r * y[0] - y[1] - y[0] * y[2]
    z_dot = y[0] * y[1] - b * y[2]
    return x_dot, y_dot, z_dot
