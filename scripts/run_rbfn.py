#%%
from jax.config import config

config.update("jax_log_compiles", 1)
# config.update("jax_debug_nans", True)

import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datagen.diffeq import gen_diffeq, vanderpol
from jax.experimental.optimizers import adam
from matplotlib.animation import FuncAnimation
from models import kernels
from models.rbfn import RBFN
from scipy.integrate import solve_ivp

from scripts.visualize import draw_vec_bg

sns.set()

t, y_true = gen_diffeq(vanderpol, n_points=2500, discard=500)

key = jax.random.PRNGKey(4)
n_rbf = 15
params = {
    "W": (W := 0.1 * jax.random.normal(key, shape=(n_rbf, 2))),
    "τ": (τ := np.abs(jax.random.normal(key))),
    "c": (c := jax.random.normal(key, shape=(n_rbf, 2))),
    "σ": (σ := np.ones(n_rbf) * 2),
}

noise = jax.random.normal(key, shape=y_true.shape) * 0.0
y = y_true + noise

#%% Online run
def train(net, x):
    for i in range(300):
        mse = net.step_online(x)
        if i % 10 == 0:
            print(i, mse)
    return net


def predict(t, x):
    x = x[np.newaxis, :]
    return net.g(x).flatten()


u = y[0:2000]  # Actual training data.
net = RBFN(kernels.linear, params, adam(2e-2))
train(net, u)
pred = solve_ivp(predict, (0, 1000), (2.0, 2.0), rtol=1e-6, max_step=1.0)

#%%
def draw(ax, u, net):
    """
    Draws - generated vector field (background, thin arrows),
          - kernel position and their associated vectors (blue dot, thick arrows),
          - training data (green arrow),
    """
    vec = np.diff(u[: net.t + 1], axis=0)
    _, im = draw_vec_bg(ax, net, n_points=20, lim=(-4, 4), minlength=0.5, width=0.002, alpha=0.4, headwidth=4)
    ax.quiver(*u[: net.t].T, *vec.T, angles="xy", scale_units="xy", scale=1.0, alpha=0.5, color="green", label="Train")
    ax.plot(*net.params["c"].T, "oC0", markeredgewidth=0.5, ms=1.0)
    ax.quiver(*net.params["c"].T, *net.params["W"].T, alpha=0.8, width=0.005)


fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
# ax.plot(*pred['y'], '--', alpha=0.7, linewidth=2)  # Draw prediction (dashed line).
draw(ax, u, net)
plt.tight_layout()

#%% Animation
fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
net = RBFN(kernels.linear, params, adam(2e-2))

u = y[::10]


def animate(i):
    ax.clear()
    mse = net.step_online(u)
    draw(ax, u, net)
    ax.set_title(f"Step: {net.t:3d}, MSE: {mse:.4f}")
    if i % 10 == 0:
        print(i, mse)


anim = FuncAnimation(fig, animate, frames=90, blit=False)  # TODO: Make blit version.
fig.tight_layout()
anim.save("myAnimation.gif", writer="imagemagick", fps=5)

#%% Kernel Comparison
kers = {"rbf": kernels.rbf, "matern32": kernels.matern32, "matern52": kernels.matern52}
nets = {k: RBFN(v, params, adam(1e-2)) for k, v in kers.items()}
[train(net, y) for net in nets.values()]

#%%
fig, axs = plt.subplots(figsize=(10, 4), dpi=300, ncols=3)
axs = axs.flatten()
vec = np.diff(u, axis=0)

for i, (name, net) in enumerate(nets.items()):
    _, im = draw_vec_bg(axs[i], net, lim=(-4, 4), y=u, show_gnd=False)
    axs[i].quiver(*u[:-1].T, *vec.T, angles="xy", scale_units="xy", scale=1.0, alpha=0.5, color="green")
    # ax.plot(*y[:100].T, "-g", alpha=0.8)
    # ax.plot(*pred['y'], '--', alpha=0.7, linewidth=2)
    axs[i].set_title(name)

plt.tight_layout()
# fig.colorbar(im).set_label("Vector Angle")

# %%
