#%%
from functools import partial
from typing import Callable

import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import numpy as onp
import seaborn as sns
from datagen.diffeq import gen_diffeq, lorenz, pca, vanderpol
from jax import jit
from jax.api import value_and_grad
from jax.config import config
from jax.experimental.optimizers import adam
from jax.interpreters.xla import DeviceArray
from jax.lax import scan
from models import kernels
from models.rbfn import RBFN
from scipy.integrate import solve_ivp

from scripts.visualize import draw_vec_bg

sns.set()

config.update("jax_log_compiles", 1)
config.update("jax_debug_nans", True)


def gen_grid(m=4, n=4):
    G = nx.grid_2d_graph(m, n)
    points = np.array(G.nodes)
    keys = {x: i for i, x in enumerate(list(G.nodes))}

    neighbors = -1 * onp.ones((points.shape[0], 4), dtype=int)
    n_neighbor = -1 * onp.ones(points.shape[0])
    for i, point in enumerate(G.nodes):
        ed = [keys[edge[1]] for edge in G.edges(point)]
        neighbors[i, : len(ed)] = ed
    return points, np.array(neighbors), np.array(n_neighbor)


m, n = 8, 8
points, nb, n_nb = gen_grid(m, n)
points = (points - np.mean(points, axis=0)) * 0.8
n_rbf = m * n


class Gravity(RBFN):
    """
    A variant that 
    """
    def __init__(self, ker: Callable, params, optimizer: tuple[Callable, ...], points) -> None:
        assert {"W", "τ", "σ"} <= params.keys()
        assert params["W"].shape[0] == params["σ"].size
        assert np.all(params["τ"] > 0) and np.all(params["σ"] > 0)

        self.init_params, self.opt_update, self.get_params = optimizer
        self.opt_update = jit(self.opt_update)
        self.opt_state = self.init_params(params)

        self.ker = ker
        self._obj = self._mse_vgrad = jit(value_and_grad(self._mse, argnums=2), static_argnums=0)
        self.gravity_grad = jit(value_and_grad(self.gravity, argnums=1))
        self.c = self.init_params(points)

        self.i = 0

    @staticmethod
    def _mse(ker: Callable, x: DeviceArray, p: dict[str, DeviceArray], c):
        "||g(x_{t-1}) + x_{t-1} - x_t||²"
        return np.mean(np.square(Gravity._g(ker, x[:-1], p, c) + x[:-1] - x[1:]))

    def g(self, x):
        return self._g(self.ker, x, self.params, self.get_params(self.c))

    @staticmethod
    @partial(jit, static_argnums=0)
    def _g(ker, x, p, c):
        W, τ, σ = p["W"], p["τ"], p["σ"]
        return ker(x, c, σ) @ W - np.exp(-(τ**2)) * x  # (4)

    def step(self, x, loop=3):
        for _ in range(loop):
            value, grads = self._obj(self.ker, x, self.params, self.get_params(self.c))
            self.opt_state = self.opt_update(self.i, grads, self.opt_state)
            val, ggrad = self.gravity_grad(x, self.get_params(self.c))
            self.c = self.opt_update(self.i, ggrad, self.c)
            self.i += 1
        return value

    @staticmethod
    def gravity(x, c):
        """
        Pull kernels toward data points by summing all the energy between each kernel and all points.
        """
        return 0.01 * np.sum(scan(Gravity.f, c, x)[1])

    @staticmethod
    def f(c, x):
        """
        c is self.c - location of each point (unchanged) (behaves like an argument here).
        x is observation.
        Check https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html for info.
        """
        return c, np.sum(Gravity.potential(np.linalg.norm(x - c, axis=1)))

    @staticmethod
    def potential(r):
        """Something similar to Lennard-Jones"""
        return 1 / np.sqrt(r) + np.log(r)


t, y_true = gen_diffeq(vanderpol, x0=(0.1, 0.1))


key = jax.random.PRNGKey(4)

params = {
    "W": (W := 0.1 * jax.random.normal(key, shape=(n_rbf, 2))),
    "τ": (τ := 3.0),
    "σ": (σ := np.ones(n_rbf) * 2),
}

#%%
noise = jax.random.normal(key, shape=y_true.shape) * 0.01
y = y_true + noise
u = y[1100:2400:5]
plt.plot(*u.T)
plt.scatter(*points.T)

#%%
def predict(t, x):
    x = x[np.newaxis, :]
    return net.g(x).flatten()


net = Gravity(kernels.rbf, params, adam(2e-2), points)

#%%
def train(net, x):
    for i in range(100):
        value = net.step(x)
        if i % 10 == 0:
            print(i, value)
    return net


train(net, u)
pred = solve_ivp(predict, (0, 1000), (2.0, 2.0), rtol=1e-6, max_step=1.0)
#%%
fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
vec = np.diff(u, axis=0)
p = net.params
_, im = draw_vec_bg(ax, net, n_points=20, lim=(-4, 4), minlength=0.5, width=0.002, alpha=0.4, headwidth=4)
ax.quiver(
    *u[:-1].T, *vec.T, angles="xy", scale_units="xy", scale=1.0, alpha=0.5, color="green", label="Train"
)

c = net.get_params(net.c)
# Grid
ax.scatter(*c.T, color="C4")
for i in range(n_rbf):
    for j in nb[i]:
        if j > -1:
            ax.plot(*c[np.array((i, j))].T, "C4", linewidth=1, alpha=0.5)
ax.quiver(*c.T, *net.params["W"].T, alpha=0.8)

ax.plot(*pred["y"], label="Predict")
ax.set_title("Linear kernel fit")
plt.legend()
# ax.plot(*y[:100].T, "-g", alpha=0.8)
# ax.plot(*pred['y'], '--', alpha=0.7, linewidth=2)

# %%
