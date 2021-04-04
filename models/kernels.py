from functools import wraps

import jax.numpy as np
from jax.numpy import exp, sqrt
from jax.numpy.linalg import norm

# From https://docs.pymc.io/api/gp/cov.html.


def normalize(func):
    @wraps(func)
    def wrapper(*args, ϵ=1e-7, **kwargs):
        res = func(*args, **kwargs)
        return res / (ϵ + np.sum(res, axis=1, keepdims=True))

    return wrapper


@normalize
def rbf(x, c, σ):
    res = list()
    for i in range(σ.size):
        res.append(exp(-norm(x - c[i], axis=1)**2 / (2*σ[i]**2)))
    return np.vstack(res).T


@normalize
def matern32(x, c, σ):
    res = list()
    for i in range(σ.size):
        u = sqrt(3) * norm(x - c[i], axis=1)
        res.append((1 + u/σ[i]) * exp(-u/σ[i]))
    return np.vstack(res).T


@normalize
def matern52(x, c, σ):
    res = list()
    for i in range(σ.size):
        u = sqrt(5) * norm(x - c[i], axis=1)
        res.append((1 + u/σ[i] + u**2 / (3 * σ[i]**2)) * exp(-u/σ[i]))
    return np.vstack(res).T


# def matern32(x, c, σ, ϵ=1e-7):
#     res = list()
#     for i in range(σ.size):
#         res.append(np.exp(-np.linalg.norm(x - c[i], axis=1) ** 2 / (2 * σ[i] ** 2)))
#     res = np.vstack(res).T
#     return res / (ϵ + np.sum(res, axis=1, keepdims=True))
