# Converts fly body position into an egocentric coordinate system.

# From https://github.com/murthylab/sleap-notebooks/blob/master/Analysis_examples.ipynb
# Data source:
# wget -O predictions.analysis.h5 https://github.com/murthylab/sleap-notebooks/raw/master/analysis_example/predictions.analysis.h5

#%%
import h5py
import numpy as np
from scipy.interpolate import interp1d


def fill_missing(Y, kind="linear"):
    """
    Fills missing values independently along each dimension after the first.
    Since some of the positions are NaNs.
    """
    # Store initial shape.
    initial_shape = Y.shape
    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))
    
    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]
        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)
        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)
        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])
        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)
    return Y


filename = "../predictions.analysis.h5"
with h5py.File(filename, "r") as f:
    dset_names = list(f.keys())
    locations = f["tracks"][:].T
    node_names = [n.decode() for n in f["node_names"][:]]

print("===filename===", filename, "", sep="\n")
print("===HDF5 datasets===", dset_names, "", sep="\n")
print("===locations data shape===", locations.shape, "", sep="\n")
print("===nodes===")
for i, name in enumerate(node_names):
    print(f"{i}: {name}")
print()

locations = fill_missing(locations)
#%% [markdown]

# ## From the authors:
# In our example file, the shape of the locations matrix (the `tracks` dataset) is (3000, 13, 2, 2) **after it is transposed** (with the `.T`). We transpose the data when loading it in Python; no transpose is needed when using MATLAB. This is because Python and MATLAB expect matrices to be stored differently.
# Here's what each dimension of the matrix means:
# - 3000: the number of frames;
# - 13: the number of nodes in the skeleton (we've also loaded and displayed the `node_names` dataset with the names of these 13 nodes);
# - 2: for the x and y coordinates;
# - 2: the number of distinct animal identities which were found (we have 2 flies in the video clip and they were tracked perfectly, so we ended up with exactly 2 track, but there may be more tracks than animals if tracking didn't work as well).

# ## Note:
# Swapping animal identity and xy coordinates for convenience.

# %% Convert to egocentric coordinate and rotate.

u = locations
center = np.mean(u[:, 1:3, :, :], axis=1, keepdims=True)

ang_vec = u[:, [1], :, :] - u[:, [2], :, :]
angle = np.arctan2(ang_vec[:, :, 0, :], ang_vec[:, :, 1, :])

centered = np.swapaxes(u - center, -1, -2)[..., np.newaxis]
rot = np.stack([  # Rotation matrix.
        np.stack([np.cos(angle), -np.sin(angle)], axis=-1), 
        np.stack([np.sin(angle), np.cos(angle)], axis=-1)
    ], axis=-2)

roted = np.einsum("...ij,...jk->...ik", rot, centered).squeeze()
# %% Plot body to confirm correct transformation.
import matplotlib.pyplot as plt


def plot_part(i, **kwargs):
    plt.plot(roted[:, i, 0, 0], roted[:, i, 0, 1], **kwargs)

plot_part(0, label="head")
plot_part(1, label="thorax")
plot_part(2, label="abdomen")
plot_part(7, label="midlegL")
plot_part(8, label="midlegR")
plt.legend()
    
from scipy.stats import zscore
# %% Process fly 0.
from sklearn.decomposition import PCA

fly0 = roted[:, :, 0, :]

pca = PCA(2)
u = zscore(pca.fit_transform(fly0.reshape([3000, -1])), axis=0)

# %%
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

key = jax.random.PRNGKey(4)
n_rbf = 40
params = {
    "W": (W := 0.1 * jax.random.normal(key, shape=(n_rbf, 2))),
    "τ": (τ := np.abs(jax.random.normal(key))),
    "c": (c := jax.random.normal(key, shape=(n_rbf, 2))),
    "σ": (σ := 2 * np.ones(n_rbf)),
}

def train(net, x):
    for i in range(200):
        mse = net.step(x)
        if i % 10 == 0:
            print(i, mse)
    return net


def predict(t, x):
    x = x[np.newaxis, :]
    return net.g(x).flatten()


net = RBFN(kernels.linear, params, adam(2e-2))
train(net, u)
# pred = solve_ivp(predict, (0, 1000), (2.0, 2.0), rtol=1e-6, max_step=1.0)
# %%
def draw(ax, u, net, online=False):
    """
    Draws - generated vector field (background, thin arrows),
          - kernel position and their associated vectors (blue dot, thick arrows),
          - training data (green arrow),
    """
    idx = net.t if online else u.shape[0] - 1
    vec = np.diff(u[: idx + 1], axis=0)
    _, im = draw_vec_bg(ax, net, n_points=20, lim=(-4, 4), minlength=0.5, width=0.002, alpha=0.4, headwidth=4)
    ax.quiver(*u[: idx].T, *vec.T, angles="xy", scale_units="xy", scale=1.0, alpha=0.5, color="green", label="Train")
    ax.plot(*net.params["c"].T, "oC0", markeredgewidth=0.5, ms=1.0)
    ax.quiver(*net.params["c"].T, *net.params["W"].T, alpha=0.8, width=0.005)


fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
# ax.plot(*pred['y'], '--', alpha=0.7, linewidth=2)  # Draw prediction (dashed line).
draw(ax, u, net)
plt.tight_layout()
# %%
