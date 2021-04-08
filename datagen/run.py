#%%
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.decomposition import PCA

from datagen import plots
from datagen.diffeq import lorenz

# %%


def random_proj(initial_dim: int, dim: int, seed=4):
    """
    Generate random Gaussian matrix with normalized columns.
    """
    rand = np.random.default_rng(seed)
    t = rand.normal(0, 1, size=(dim, initial_dim))
    return (t / np.sum(t, axis=0)).T


def random_rotation(initial_dim: int, dim: int, θ: float, seed=4):
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


# fmt: off
def gen_data_diffeq(f: Callable, projection: Callable, *, t, x0: np.ndarray, dim: int,
                    ivp_kwargs: Optional[dict] = None, proj_kwargs: Optional[dict] = None,
                    noise: Optional[str] = None, noise_kwargs: Optional[dict] = None,
                    seed=41):  # fmt: skip 
# fmt: on

    """Data generation pipeline
    Solve dynamical system, project into higher dimension, add noise.
    
    Args:
        f (Callable): Diff function to solve.
        projection (Callable): Proj function. Must have signature (initial_dim, dim, **kwargs) and output matrix (initial_dim × dim).
        t: Time for diff eq. See solve_ivp documentation.
        x0 (np.ndarray): Starting point.
        dim (int): Projected number of dimensions.
        ivp_args (Optional[dict]): Args to solve_ivp. Defaults to None.
        noise (Optional[str]): Name of noise function from https://numpy.org/doc/stable/reference/random/generator.html
        noise_params (Optional[dict]): Params to noise function. Defaults to None.
        seed (int): Defaults to 41.

    Returns:
        t, y (t × initial_dim), projected (t × dim)
    """
    if ivp_kwargs is None:
        ivp_kwargs = dict()
    if proj_kwargs is None:
        proj_kwargs = dict()
    if noise_kwargs is None:
        noise_kwargs = dict()

    ivp = solve_ivp(f, t, x0, rtol=1e-6, **ivp_kwargs)

    y = ivp["y"].T  # (t × dim)
    proj = projection(y.shape[1], dim, **proj_kwargs)
    projed = y @ proj

    if noise is not None:
        rand = np.random.default_rng(seed)
        projed += getattr(rand, noise)(**noise_kwargs, size=projed.shape)

    return ivp["t"], y, projed


# %% Example
t, y, projed = gen_data_diffeq(lorenz, random_proj, t=(0, 50), x0=[0.1, 0.1, 0.1], dim=100, noise="normal", noise_kwargs={"loc": 0, "scale": 1},)

pcaed = PCA(n_components=3).fit_transform(projed)
plots.plot3d_color(pcaed, t)
#%%
fig, axs = plt.subplots(ncols=2)
plots.plot_color(y[:, 0], y[:, 1], t, axs[0])
plots.plot_color(pcaed[:, 0], pcaed[:, 1], t, axs[1])
plt.tight_layout()

# %%
from streamingSVD.ssSVD import get_ssSVD

Qtcoll, Scoll, Qcoll = get_ssSVD(projed, 3, 10, 10, 5)
plots.plot_color(Qcoll[:, 0, 0], Qcoll[:, 1, 0], t)

plt.show()

# %%
