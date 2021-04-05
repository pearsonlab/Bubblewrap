#%%
from functools import partial
from typing import Callable

import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import numpy as onp
import seaborn as sns
from jax import jit
from jax.api import value_and_grad
from jax.config import config
from jax.experimental.optimizers import adam
from jax.interpreters.xla import DeviceArray
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
    def __init__(self, ker: Callable, params, optimizer: tuple[Callable, ...],
        *, params_spr: dict[str, float], nb: DeviceArray
    ):
        super().__init__(ker, params, optimizer)
        assert nb.shape[0] == self.params["σ"].shape[0]
        self.nb = nb
        
        assert {"k", "l0"} <= params_spr.keys()
        self.p_spr = params_spr

        self._obj = partial(
            jit(value_and_grad(self._mse_spring, argnums=2), static_argnums=0),
            k=self.p_spr["k"],
            l0=self.p_spr["l0"],
        )

    @staticmethod
    def _mse_spring(kern: Callable, x: DeviceArray, p: dict[str, DeviceArray], **kwargs):
        return Spring._mse(kern, x, p) + Spring.spring_energy(p["c"], nb, **kwargs)

    @staticmethod
    def spring_energy(coords, neighbors, k=1.0, l0=1.0):
        arr = coords[neighbors]  # (points × n_neighbors × dim)
        mask = (neighbors > -1).astype(np.float32)[..., np.newaxis]

        centered = (arr - coords[:, np.newaxis, :]) * mask + 1e-7  # Zero out mask and prevent sqrt(-0).
        ℓ = np.linalg.norm(centered, axis=2)

        return 0.5 * k * np.sum((ℓ - l0) ** 2)  # energy


t, y_true = gen(vanderpol, x0=(0.1, 0.1))

m, n = 5, 5
points, nb, n_nb = gen_grid(m, n)
points = (points - np.mean(points, axis=0)) * 1.0

key = jax.random.PRNGKey(4)
n_rbf = m * n
params = {
    "W": (W := 0.1 * jax.random.normal(key, shape=(n_rbf, 2))),
    "τ": (τ := 3.0),
    "c": (c := points),
    "σ": (σ := np.ones(n_rbf) * 2),
}

params_spr = {
    "k": 0.005,
    "l0": 1.,
}
#%%
noise = jax.random.normal(key, shape=y_true.shape) * 0.1
y = y_true + noise
u = y[1100:2400:5]
plt.plot(*u.T)
plt.scatter(*points.T)

#%%
def train(net, x):
    for i in range(500):
        value = net.step(x)
        if i % 10 == 0:
            print(i, value)
    return net


net = Spring(kernels.linear, params, adam(2e-2), params_spr=params_spr, nb=nb)
train(net, y)


def predict(t, x):
    x = x[np.newaxis, :]
    return net.g(x).flatten()


pred = solve_ivp(predict, (0, 1000), (2.0, 2.0), rtol=1e-6, max_step=1.0)
#%%
fig, ax = plt.subplots(figsize=(10, 4), dpi=200)

vec = np.diff(u, axis=0)
p = net.params
_, im = draw(ax, net, lim=(-4, 4))
ax.quiver(
    *u[:-1].T, *vec.T, angles="xy", scale_units="xy", scale=1.0, alpha=0.5, color="green", label="Train"
)


# Grid
# ax.scatter(*net.params["c"].T, color="C4")
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
