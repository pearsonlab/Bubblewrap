from typing import Callable

import jax.numpy as np
import numpy as onp
from scipy.integrate import solve_ivp
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def lorenz(t, y: np.ndarray, s=10., r=28., b=2.667):  # 3D
    x_dot = s * (y[1] - y[0])
    y_dot = r * y[0] - y[1] - y[0] * y[2]
    z_dot = y[0] * y[1] - b * y[2]
    return x_dot, y_dot, z_dot


def vanderpol(t, f, mu=1.):  # 2D
    x, y = f
    x_dot = y
    y_dot = mu * (1 - x ** 2) * y - x
    return x_dot, y_dot


def gen(func: Callable, x0=(0.1, 0.1), *,
        t_lim=50.0, n_points=4000, discard=0, rtol=1e-6, max_step=1.0, **kwargs):
    """
    When max_step == 1, t_lim == n_points
    """
    t = onp.linspace(0., t_lim, n_points, dtype=np.float64)
    sol = solve_ivp(func, [0., t_lim], x0, t_eval=t, rtol=rtol, max_step=max_step, **kwargs)
    return np.array(t)[discard:], np.array(sol.y.T)[discard:]


def pca(y, dim=2):
    return np.array(PCA(dim).fit_transform(y))


def kmeans(y, n_rbf: int):
    # Original RBF implementation uses K-means initialization.
    km = KMeans(n_rbf).fit(y)
    c = km.cluster_centers_
    dist = pdist(c)  # Pairwise distance
    return c, np.mean(dist)
