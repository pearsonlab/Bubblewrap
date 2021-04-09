#%%
from jax.config import config

config.update("jax_log_compiles", 1)
config.update("jax_debug_nans", True)

import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datagen.diffeq import gen_diffeq, lorenz, pca, vanderpol
from field.rbfn_spring import RBFNSpring, gen_grid
from jax.experimental.optimizers import adam
from models import kernels

from scripts.visualize import draw_vec_bg

t, y_true = gen_diffeq(lorenz, x0=(0.1, 0.1, 0.1))
y_true = pca(y_true)

sns.set()

m, n = 4, 4
points, nb, n_nb = gen_grid(m, n)
points = (points - np.mean(points, axis=0)) * 15.

key = jax.random.PRNGKey(4)
n_rbf = m * n
params = {
    "W": (W := 0.1 * jax.random.normal(key, shape=(n_rbf, 2))),
    "τ": (τ := 3.0),
    "c": (c := points),
    "σ": (σ := np.ones(n_rbf) * 2)
}

params_spr = {
    "k": 0.001,
    "l0": 15.,
}
#%%

noise = jax.random.normal(key, shape=y_true.shape) * 2
y = y_true + noise
u = y[2100:2400]
plt.plot(*u.T)
plt.plot(*points.T)

#%%
def train(net, x):
    for i in range(1000):
        value = net.step(x, nb=nb)
        if i % 10 == 0:
            print(i, value)
    return net


net = RBFNSpring(kernels.linear, params, adam(2e-2), params_spr=params_spr, nb=nb)
train(net, y)


# %%
