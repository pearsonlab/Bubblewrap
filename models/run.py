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
from models.rbfn import RBFN
from models.visualize import draw

sns.set()


t = np.linspace(0, 1000, 5000)
ivp = solve_ivp(vanderpol, (0, 1000), (0.1, 0.1), rtol=1e-6)
y_true = ivp["y"].T[1000:]

key = jax.random.PRNGKey(4)
n_rbf = 15
params = {
    "W": (W := 0.1*jax.random.normal(key, shape=(n_rbf, 2))),
    "τ": (τ := np.abs(jax.random.normal(key))),
    "c": (c := jax.random.normal(key, shape=(n_rbf, 2))),
    "σ": (σ := np.ones(n_rbf) * 2),
}

noise = jax.random.normal(key, shape=y_true.shape) * 0.01
y = y_true + noise

def train(net, y):
    for i in range(1000):
        value = net.step(y[:100])
        if i % 10 == 0:
            print(i, value)
    return net

#%% Normal run
u = y[:100]
net = RBFN(kernels.rbf, params, adam(2e-2))
train(net, u)

#%%
fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
_, im = draw(ax, net, lim=(-4, 4), y=u, show_gnd=False)

vec = np.diff(u, axis=0)
ax.quiver(*u[:-1].T, *vec.T, angles='xy', scale_units='xy', scale=1., alpha=0.5, color='green')
# ax.plot(*y[:100].T, "-g", alpha=0.8)
# ax.plot(*pred['y'], '--', alpha=0.7, linewidth=2)

plt.tight_layout()

#%% Kernel Comparison
kers = {'rbf': kernels.rbf, 'matern32': kernels.matern32, 'matern52': kernels.matern52}
nets = {k: RBFN(v, params, adam(1e-2)) for k, v in kers.items()}
[train(net, y) for net in nets.values()]

#%%
fig, axs = plt.subplots(figsize=(10, 4), dpi=300, ncols=3)
axs = axs.flatten()
vec = np.diff(u, axis=0)

for i, (name, net) in enumerate(nets.items()):
    _, im = draw(axs[i], net, lim=(-4, 4), y=u, show_gnd=False)
    axs[i].quiver(*u[:-1].T, *vec.T, angles='xy', scale_units='xy', scale=1., alpha=0.5, color='green')
    # ax.plot(*y[:100].T, "-g", alpha=0.8)
    # ax.plot(*pred['y'], '--', alpha=0.7, linewidth=2)
    axs[i].set_title(name)

plt.tight_layout()
# fig.colorbar(im).set_label("Vector Angle")

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
