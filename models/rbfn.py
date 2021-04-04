#%%
"""
An implementation of `Interpretable Nonlinear Dynamic Modeling of Neural Trajectories`
Yuan Zhao, Il Memming Park, NIPS 2016

Equations are exact matches to those in the paper.
Generate data from a van der pol oscillator, fit with MSE, and draw vector field.
Takes ~5 ms to run per step on a 4 GHz Coffee Lake CPU.

"""
    
from functools import partial
from typing import Callable

from jax.config import config
from jax.interpreters.xla import DeviceArray

config.update("jax_log_compiles", 1)
config.update("jax_debug_nans", True)

import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as onp
import seaborn as sns
from datagen.models import vanderpol
from datagen.plots import plot_color
from jax import jit
from jax.api import grad, value_and_grad
from jax.experimental.optimizers import adam
from scipy.integrate import solve_ivp
from sklearn.decomposition import PCA

from models import kernels

sns.set()


t = np.linspace(0, 1000, 5000)
ivp = solve_ivp(vanderpol, (0, 1000), (0.1, 0.1), rtol=1e-6)
y_true = ivp["y"].T[1000:]


#%%
class RBFN:
    def __init__(self, ker: Callable, params, optimizer: tuple[Callable, ...]) -> None:
        assert {"W", "τ", "c", "σ"} <= params.keys()
        assert params["W"].shape == params["c"].shape
        assert params["W"].shape[0] == params["σ"].size
        assert np.all(params["τ"] > 0) and np.all(params["σ"] > 0)
        
        self.init_params, self.opt_update, self.get_params = optimizer
        self.opt_update = jit(self.opt_update)
        self.opt_state = self.init_params(params)
        
        self.ker = ker
        self._mse_vgrad = jit(value_and_grad(RBFN._mse, argnums=2), static_argnums=0)
        self.i = 0
        
    @property
    def params(self):
        return self.get_params(self.opt_state)
    
    def g(self, x):
        return self._g(self.ker, x, **self.params)
    
    def step(self, x, loop=3):
        for _ in range(loop):
            value, grads = self._mse_vgrad(self.ker, x, self.params)
            self.opt_state = self.opt_update(self.i, grads, self.opt_state)
        
        self.i += 1
        return value

    @staticmethod
    @partial(jit, static_argnums=0)
    def _g(kern, x, W, τ, c, σ):
        return kern(x, c, σ) @ W - np.exp(-(τ**2)) * x  # (4)

    @staticmethod
    def _mse(kern: Callable, x: DeviceArray, p: dict[str, DeviceArray]):
        return np.mean(
            np.square(RBFN._g(kern, x[:-1], p["W"], p["τ"], p["c"], p["σ"]) + x[:-1] - x[1:])
        )


key = jax.random.PRNGKey(4)
n_rbf = 15
params = {
    "W": (W := jax.random.normal(key, shape=(n_rbf, 2))),
    "τ": (τ := np.abs(jax.random.normal(key))),
    "c": (c := jax.random.normal(key, shape=(n_rbf, 2))),
    "σ": (σ := np.ones(n_rbf) * 2),
}

noise = jax.random.normal(key, shape=y_true.shape) * 0.3
y = y_true + noise
# net = RBFN(kernels.rbf, params, adam(2e-2))


#%%
def gen_vec(net, n_points=20, lim=(-40, 40)):
    X = np.linspace(*lim, n_points)
    U, V = onp.meshgrid(X, X)
    points = np.vstack((U.flatten(), V.flatten())).T

    vec = net.g(points)
    U, V = vec[:, 0].reshape((n_points, n_points)), vec[:, 1].reshape((n_points, n_points))
    return X, U, V

def draw(ax, net, n_points=20, lim=(-40, 40), y=None, show_gnd=True):
    X, U, V = gen_vec(net, n_points, lim)
    mag = np.sqrt(U ** 2 + V ** 2)
    vel = np.linalg.norm(np.diff(y, axis=0), axis=1)
    
    if show_gnd:
        plot_color(*y.T, t=vel, ax=ax, cmap="magma_r", alpha=1)
    ax.quiver(X, X, U, V, alpha=0.8)
    
    X, U, V = gen_vec(net, 300, lim)
    θ = np.arctan2(U, V)
    im = ax.imshow(np.flipud(θ.reshape((300, 300))), extent=[*lim, *lim], cmap="twilight")
    
    ax.set_aspect("equal")
    ax.set_title("RBF network fit. Color indicates speed/magnitude.")
    ax.grid(0)
    return ax, im



#%%
kers = {'rbf': kernels.rbf, 'matern32': kernels.matern32, 'matern52': kernels.matern52}
nets = {k: RBFN(v, params, adam(2e-2)) for k, v in kers.items()}

def train(net, y):
    for i in range(500):
        value = net.step(y[:100])
        if i % 10 == 0:
            print(i, value)
    return net

[train(net, y) for net in nets.values()]

#%%
fig, axs = plt.subplots(figsize=(10, 4), dpi=300, ncols=3)
axs = axs.flatten()
u = y[:100]
vec = np.diff(u, axis=0)

for i, (name, net) in enumerate(nets.items()):
    _, im = draw(axs[i], net, lim=(-4, 4), y=y[:100], show_gnd=False)
    axs[i].quiver(*u[:-1].T, *vec.T, angles='xy', scale_units='xy', scale=1., alpha=0.5, color='green')

    # ax.plot(*y[:100].T, "-g", alpha=0.8)
    # ax.plot(*pred['y'], '--', alpha=0.7, linewidth=2)
    axs[i].set_title(name)

plt.tight_layout()
# fig.colorbar(im).set_label("Vector Angle")


#%%
def predict(t, y):
    y = y[np.newaxis, :]
    return net.g(y).flatten()

predict(0, np.array([0.1, 0.1]))
pred = solve_ivp(predict, (0, 100), (2.,2.), rtol=1e-6, max_step=0.1)


#%%extent=(*lim, *lim)
# from sklearn.decomposition import PCA

# y = PCA(2).fit_transform(y)
# from sklearn.cluster import KMeans
# km = KMeans(n_rbf).fit(y)
# c = km.cluster_centers_
# x = []
# for i in range(n_rbf):
#     for j in range(i, n_rbf):
#         x.append(np.sqrt(np.mean((c[i] - c[j])**2)))
# σ = sum(x) / len(x)
#%%

# plt.quiver(*u[:-1].T, *vec.T, angles='xy', scale_units='xy', scale=1.)

# %%
