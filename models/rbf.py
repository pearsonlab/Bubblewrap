#%%
from jax.config import config

config.update("jax_log_compiles", 1)

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

sns.set()


t = np.linspace(0, 1000, 5000)
ivp = solve_ivp(vanderpol, (0, 1000), (0.1, 0.1), rtol=1e-6)
y = ivp["y"].T[1000:]


#%%
class RBF:
    def __init__(self) -> None:
        # assert ["W", "τ", "c", "σ"] <= params.keys()
        # assert params["W"].shape == params["c"].shape
        # assert params["W"].shape[0] == params["σ"].size
        # assert params["τ"] > 0 and params["σ"] > 0

        # self.params = params
        self.mse_grad = jit(grad(self.mse, argnums=1))
        self.mse_vgrad = jit(value_and_grad(self.mse, argnums=1))

    def mse(self, x):
        return self._mse(x, self.params)

    @staticmethod
    def φ(x, c, σ, ϵ=1e-7):
        res = list()
        for i in range(σ.size):
            res.append(np.exp(-np.linalg.norm(x - c[i], axis=1) ** 2 / (2 * σ[i] ** 2)))
        res = np.vstack(res).T
        return res / (ϵ + np.sum(res, axis=1, keepdims=True))

    @staticmethod
    def g(x, W, τ, c, σ):
        return RBF.φ(x, c, σ) @ W - np.exp(-(τ ** 2)) * x  # (4)

    @staticmethod
    def mse(x, p):
        return np.mean(
            np.square(RBF.g(x[:-1], p["W"], p["τ"], p["c"], p["σ"]) + x[:-1] - x[1:])
        )


key = jax.random.PRNGKey(4)
rbf = RBF()
init_params, opt_update, get_params = adam(2e-2)
opt_update = jit(opt_update)
n_rbf = 20
params = {
    "W": (W := jax.random.normal(key, shape=(n_rbf, 2))),
    "τ": (τ := np.abs(jax.random.normal(key))),
    "c": (c := 20 * jax.random.normal(key, shape=(n_rbf, 2))),
    "σ": (σ := np.ones(n_rbf) * 5),
}

def step(step, opt_state, loop=3):
    for _ in range(loop):
        grads = rbf.mse_grad(y, get_params(opt_state))
        opt_state = opt_update(step, grads, opt_state)
    return opt_state


opt_state = init_params(params)
for i in range(200):
    opt_state = step(i, opt_state)

#%%
def draw(params, n_points=20, lim=(-40, 40)):
    lim = np.linspace(*lim, n_points)
    U, V = onp.meshgrid(lim, lim)
    points = np.vstack((U.flatten(), V.flatten())).T

    vec = RBF.g(points, **params)
    U, V = vec[:, 0].reshape((n_points, n_points)), vec[:, 1].reshape((n_points, n_points))
    mag = np.sqrt(U ** 2 + V ** 2)
    vel = np.linalg.norm(np.diff(y, axis=0), axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    plot_color(*y[1:].T, t=vel, ax=ax)
    ax.quiver(lim, lim, U, V, mag)
    ax.set_aspect("equal")
    ax.set_title("RBF network fit. Color indicates speed/magnitude.")
    return fig, ax

draw(get_params(opt_state))


#%%
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
