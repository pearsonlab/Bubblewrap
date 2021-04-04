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

def draw(ax, net, n_points=20, lim=(-40, 40), y=None, show_gnd=True):
    X, U, V = gen_vec(net, n_points, lim)
    mag = np.sqrt(U ** 2 + V ** 2)
    vel = np.linalg.norm(np.diff(y, axis=0), axis=1)
    
    if show_gnd:
        plot_color(*y.T, t=vel, ax=ax, cmap="magma_r", alpha=1)
    ax.quiver(X, X, U, V, alpha=0.8)
    
    X, U, V = gen_vec(net, 300, lim)
    θ = np.arctan2(U, V)
    im = ax.imshow(np.flipud(θ.reshape((300, 300))), extent=[*lim, *lim], cmap="twilight")
    
    ax.set_aspect("equal")
    ax.set_title("RBF network fit. Color indicates speed/magnitude.")
    ax.grid(0)
    return ax, im
