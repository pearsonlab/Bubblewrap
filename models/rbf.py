#%%
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datagen.models import vanderpol
from datagen.plots import plot_color
from jax.experimental.optimizers import adam
from scipy.integrate import solve_ivp
from sklearn.decomposition import PCA

sns.set()
n_rbf = 20

#%%
def φ(x, c, σ):
    out = np.zeros((x.shape[0], n_rbf))
    for i in range(n_rbf):
        out = jax.ops.index_update(
            out, jax.ops.index[:, i],
            np.exp(-np.linalg.norm(x - c[i], axis=1) ** 2 / (2 * σ[i])),
        )
    return out / (1e-7 + np.sum(out, axis=1, keepdims=True))


def g(x, W, τ, c, σ):
    return φ(x, c, σ) @ W - np.exp(-(τ ** 2)) * x  # (4)


def mse(x, p):
    return np.mean((g(x[:-1], p["W"], p["τ"], p["c"], p["σ"]) + x[:-1] - x[1:]) ** 2)


t = np.linspace(0, 1000, 5000)
ivp = solve_ivp(vanderpol, (0, 1000), (0.1, 0.1), rtol=1e-6)
y = ivp["y"].T[1000:]


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

#%%
init_params, opt_update, get_params = adam(2e-2)
key = jax.random.PRNGKey(4)
idxs = jax.random.choice(key, np.arange(y.shape[0]), (n_rbf,))

params = {
    "W": jax.random.normal(key, shape=(n_rbf, 2)),
    "τ": np.abs(jax.random.normal(key)),
    "c": 20 * jax.random.normal(key, shape=(n_rbf, 2)),
    "σ": np.ones(n_rbf) * 5,
}
#%%


def step(step, opt_state):
    for i in range(3):
        value, grads = jax.value_and_grad(mse, argnums=1)(y[: 2 * (step + 1)], get_params(opt_state))
        opt_state = opt_update(step, grads, opt_state)
    return value, opt_state


opt_state = init_params(params)
for i in range(200):
    value, opt_state = step(i, opt_state)
    print(i, value)


# %%
opted = get_params(opt_state)

n_points = 20
import numpy as onp

lim = np.linspace(-40, 40, n_points)
U, V = onp.meshgrid(lim, lim)
points = np.vstack((U.flatten(), V.flatten())).T

vec = g(points, **opted)
U, V = vec[:, 0].reshape((n_points, n_points)), vec[:, 1].reshape((n_points, n_points))
mag = np.sqrt(U ** 2 + V ** 2)

vel = np.linalg.norm(np.diff(y, axis=0), axis=1)
# %%
from matplotlib import rc

plt.rcParams["font.family"] = "sans-serif"


rc("text", usetex=True)
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
plot_color(*y[1:].T, t=vel, ax=ax, alpha=0.7)
ax.quiver(lim, lim, U, V, mag)
ax.set_aspect("equal")
ax.set_title("RBF network fit. Color indicates speed/magnitude.")
# %%
