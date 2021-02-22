#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.decomposition import PCA

from datagen import plots
from datagen.models import lorenz

# %%


def random_proj(initial_dim: int, dim: int, seed=4):
    """
    Generate random Gaussian matrix with normalized columns.
    """
    rand = np.random.default_rng(seed)
    t = rand.normal(0, 1, size=(dim, initial_dim))
    return (t / np.sum(t, axis=0)).T


def random_rotation(dim: int, θ: float, seed=4):
    rot = np.array(
        [
            [np.cos(θ), -np.sin(θ)],
            [np.sin(θ), np.cos(θ)],
        ]
    )
    out = np.zeros((dim, dim))
    out[:2, :2] = rot
    rand = np.random.default_rng(seed)
    q = np.linalg.qr(rand.normal(0, 1, size=(dim, dim)))[0]
    return q @ out @ q.T


# %% Example
ivp = solve_ivp(lorenz, (0, 50), [0.1, 0.1, 0.1], args=(), rtol=1e-6)
proj = random_proj(3, 100)
rand = np.random.default_rng(41)

projed = ivp["y"].T @ proj
projed += rand.normal(0, 0.1, size=projed.shape)

pcaed = PCA(n_components=3).fit_transform(projed)
plots.plot3d_color(pcaed, ivp["t"])
#%%
fig, axs = plt.subplots(ncols=2)
plots.plot_color(ivp["y"][0, :], ivp["y"][1, :], ivp["t"], axs[0])
plots.plot_color(pcaed[:, 0], pcaed[:, 1], ivp["t"], axs[1])
plt.tight_layout()

# %%
from streamingSVD.ssSVD import get_ssSVD

Qtcoll, Scoll, Qcoll = get_ssSVD(projed, 3, 10, 10, 5)
plots.plot_color(Qcoll[:, 0, 0], Qcoll[:, 1, 0], ivp["t"])

# %%
