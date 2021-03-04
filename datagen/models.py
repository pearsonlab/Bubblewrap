import numpy as np

def lorenz(t, y: np.ndarray, s=10, r=28, b=2.667):
    # 3D
    x_dot = s * (y[1] - y[0])
    y_dot = r * y[0] - y[1] - y[0] * y[2]
    z_dot = y[0] * y[1] - b * y[2]
    return x_dot, y_dot, z_dot

def vanderpol(t, f, mu=1):
    # 2D
    x, y = f 
    x_dot = y
    y_dot = mu*(1-x**2)*y - x
    return x_dot, y_dot