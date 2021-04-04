#%%
from jax.config import config

config.update("jax_log_compiles", 1)
config.update("jax_debug_nans", True)

import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datagen.models import vanderpol
from jax.experimental.optimizers import adam
from scipy.integrate import solve_ivp

from models import kernels
from models.diffeq import gen, vanderpol
from models.rbfn import RBFN
from models.visualize import draw

sns.set()

t, y_true = gen(vanderpol, n_points=2000, discard=500)

key = jax.random.PRNGKey(4)
n_rbf = 15
params = {
    "W": (W := 0.1 * jax.random.normal(key, shape=(n_rbf, 2))),
    "τ": (τ := np.abs(jax.random.normal(key))),
    "c": (c := jax.random.normal(key, shape=(n_rbf, 2))),
    "σ": (σ := np.ones(n_rbf) * 2),
}

noise = jax.random.normal(key, shape=y_true.shape) * 0.6
y = y_true + noise

#%% Normal run
def train(net, x):
    for i in range(1000):
        value = net.step(x)
        if i % 10 == 0:
            print(i, value)
    return net

def predict(t, x):
    x = x[np.newaxis, :]
    return net.g(x).flatten()

u = y[0:1000:10]
net = RBFN(kernels.linear, params, adam(2e-2))
train(net, u)
pred = solve_ivp(predict, (0, 1000), (2.,2.), rtol=1e-6, max_step=1.)

#%%
fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
_, im = draw(ax, net, lim=(-4, 4))

vec = np.diff(u, axis=0)
ax.quiver(*u[:-1].T, *vec.T, angles="xy", scale_units="xy", scale=1.0, alpha=0.5, color="green")
ax.plot(*pred['y'], '--', alpha=0.7, linewidth=2)
ax.scatter(*net.params["c"].T)
ax.quiver(*net.params["c"].T, *net.params["W"].T, alpha=0.8)
plt.tight_layout()


#%% Kernel Comparison
kers = {"rbf": kernels.rbf, "matern32": kernels.matern32, "matern52": kernels.matern52}
nets = {k: RBFN(v, params, adam(1e-2)) for k, v in kers.items()}
[train(net, y) for net in nets.values()]

#%%
fig, axs = plt.subplots(figsize=(10, 4), dpi=300, ncols=3)
axs = axs.flatten()
vec = np.diff(u, axis=0)

for i, (name, net) in enumerate(nets.items()):
    _, im = draw(axs[i], net, lim=(-4, 4), y=u, show_gnd=False)
    axs[i].quiver(*u[:-1].T, *vec.T, angles="xy", scale_units="xy", scale=1.0, alpha=0.5, color="green")
    # ax.plot(*y[:100].T, "-g", alpha=0.8)
    # ax.plot(*pred['y'], '--', alpha=0.7, linewidth=2)
    axs[i].set_title(name)

plt.tight_layout()
# fig.colorbar(im).set_label("Vector Angle")
