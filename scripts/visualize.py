import jax.numpy as np
import numpy as onp
from datagen.plots import plot_color


def gen_vec(net, n_points=20, lim=(-40, 40)):
    X = np.linspace(*lim, n_points)
    U, V = onp.meshgrid(X, X)
    points = np.vstack((U.flatten(), V.flatten())).T

    vec = net.g(points)
    U, V = vec[:, 0].reshape((n_points, n_points)), vec[:, 1].reshape((n_points, n_points))
    return X, U, V


def paint(ax, x, lim, px=300):
    return ax.imshow(np.flipud(x.reshape((px, px))), extent=[*lim, *lim], cmap="twilight")


def draw_vec_bg(ax, net, n_points=10, lim=(-40, 40), px=300, draw_mag=False, **kwargs):
    X, U, V = gen_vec(net, n_points, lim)
    ax.quiver(X, X, U, V, **kwargs)

    X, U, V = gen_vec(net, px, lim)
    mag = np.sqrt(U ** 2 + V ** 2)
    θ = np.arctan2(U, V)

    im = paint(ax, mag, lim, px=px) if draw_mag else paint(ax, θ, lim, px=px)
    ax.set_aspect("equal")
    ax.grid(0)
    return ax, im
