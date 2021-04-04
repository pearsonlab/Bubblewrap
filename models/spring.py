#%%

from functools import partial
from typing import Callable

import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import numpy as onp
import seaborn as sns
from datagen.plots import plot3d_color, plot_color
from jax import jit
from jax.api import value_and_grad
from jax.config import config
from jax.experimental.optimizers import adam
from jax.interpreters.xla import DeviceArray
from networkx.generators.lattice import grid_2d_graph
from scipy.integrate import solve_ivp

from models import kernels
from models.diffeq import gen, lorenz, pca, vanderpol
from models.rbfn import RBFN
from models.visualize import draw
sns.set()

config.update("jax_log_compiles", 1)
config.update("jax_debug_nans", True)
# config.update('jax_disable_jit', True)

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


class Spring(RBFN):
    def __init__(self, ker: Callable, params, optimizer: tuple[Callable, ...], nb: DeviceArray) -> None:
        super().__init__(ker, params, optimizer)
        self.nb = nb
        self._mse_vgrad = jit(value_and_grad(self._mse, argnums=2), static_argnums=0)

    @staticmethod
    @partial(jit, static_argnums=0)
    def _g(kern, x, W, τ, c, σ, **kwargs):
        return kern(x, c, σ) @ W  # - np.exp(-(τ**2)) * x  # (4)

    @staticmethod
    def _mse(kern: Callable, x: DeviceArray, p: dict[str, DeviceArray]):
        return np.mean(
            np.square(RBFN._g(kern, x[:-1], p["W"], p["τ"], p["c"], p["σ"]) + x[:-1] - x[1:])
        ) + 0.001 * Spring.spring_energy(p["c"], nb, l0=p["l0"])

    @staticmethod
    def spring_energy(coords, neighbors, k=1.0, l0=1.0):
        arr = coords[neighbors]  # (points × n_neighbors × dim)
        mask = (neighbors > -1).astype(np.float32)[..., np.newaxis]

        centered = (arr - coords[:, np.newaxis, :]) * mask + 1e-7  # Zero out mask and prevent sqrt(-0).
        ℓ = np.linalg.norm(centered, axis=2)

        return 0.5 * k * np.sum((ℓ - l0) ** 2)  # energy


t, y_true = gen(lorenz, x0=(0.1, 0.1, 0.1))
y_true = pca(y_true)

m, n = 6, 6
points, nb, n_nb = gen_grid(m, n)
points = (points - np.mean(points, axis=0)) * 10

key = jax.random.PRNGKey(4)
n_rbf = m * n
params = {
    "W": (W := 0.1 * jax.random.normal(key, shape=(n_rbf, 2))),
    "τ": (τ := 3.0),
    "c": (c := points),
    "σ": (σ := np.ones(n_rbf) * 2),
    "l0": 10.
    # "k": (k := 0.01)
}
#%%

noise = jax.random.normal(key, shape=y_true.shape) * 0.5
y = y_true + noise
u = y[2100:2400]


def train(net, x):
    for i in range(1000):
        value = net.step(x)
        if i % 10 == 0:
            print(i, value)
    return net


net = Spring(kernels.rbf, params, adam(2e-2), nb=nb)
train(net, y)


def predict(t, x):
    x = x[np.newaxis, :]
    return net.g(x).flatten()


pred = solve_ivp(predict, (0, 1000), (2.0, 2.0), rtol=1e-6, max_step=1.0)
#%%
fig, ax = plt.subplots(figsize=(10, 4), dpi=200)

vec = np.diff(u, axis=0)
p = net.params
_, im = draw(ax, net, lim=(-30, 30))
ax.quiver(
    *u[:-1].T, *vec.T, angles="xy", scale_units="xy", scale=1.0, alpha=0.5, color="green", label="Train"
)


# Grid
ax.scatter(*net.params["c"].T, color="C4")
for i in range(n_rbf):
    for j in nb[i]:
        if j > -1:
            ax.plot(*p["c"][np.array((i, j))].T, "C4", linewidth=1, alpha=0.5)
ax.quiver(*net.params["c"].T, *net.params["W"].T, alpha=0.8)

ax.plot(*pred["y"], label="Predict")
ax.set_title("Linear kernel fit")
plt.legend()
# ax.plot(*y[:100].T, "-g", alpha=0.8)
# ax.plot(*pred['y'], '--', alpha=0.7, linewidth=2)


#%%
