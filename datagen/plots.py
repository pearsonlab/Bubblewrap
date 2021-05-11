import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def plot3d_color(y, t, cmap="viridis", **kwargs):
    """
    Plot a 3D trajectory `y` with colors from `t`.

    Args:
        t (np.ndarray): 1D data.
        y (np.ndarray): n × 3 data.

    Returns:
        Fig, ax
    """
    y = y.T.reshape(-1, 1, 3)

    segments = np.concatenate([y[:-1], y[1:]], axis=1)

    norm = plt.Normalize(t.min(), t.max())
    lc = Line3DCollection(segments, cmap=cmap, norm=norm, alpha=0.9, **kwargs)
    lc.set_array(t)
    lc.set_linewidth(2)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    line = ax.add_collection3d(lc)

    ax.axes.set_xlim3d(
        left=y[:, :, 0].min(), right=y[:, :, 0].max()
    )  # TODO: Find a better way.
    ax.axes.set_ylim3d(bottom=y[:, :, 1].min(), top=y[:, :, 1].max())
    ax.axes.set_zlim3d(bottom=y[:, :, 2].min(), top=y[:, :, 2].max())
    return fig, ax


def plot_color(x, y, t, ax=None, colorbar=True, cmap="viridis", alpha=0.9, **kwargs):
    """
    Plot a trajectory `x` and `y` with colors from `t`.
    """
    y = np.vstack([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([y[:-1], y[1:]], axis=1)

    norm = plt.Normalize(t.min(), t.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=alpha, **kwargs)
    lc.set_array(t)
    lc.set_linewidth(2)

    if ax is None:
        fig, ax = plt.subplots()
    line = ax.add_collection(lc)
    ax.autoscale()
    if colorbar and ax is None:
        fig.colorbar(line)
    return ax
