import sys
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp
from tqdm import trange


def lorenz(t, y: np.ndarray, s=10., r=28., b=2.667):
    """
    copy & pasted here in order to avoid importing jax, which doesn't run on M1 macs
    """
    x_dot = s * (y[1] - y[0])
    y_dot = r * y[0] - y[1] - y[0] * y[2]
    z_dot = y[0] * y[1] - b * y[2]
    return x_dot, y_dot, z_dot


def vdp(t, f, mu=1.):  # 2D
    x, y = f
    x_dot = y
    y_dot = mu * (1 - x ** 2) * y - x
    return x_dot, y_dot


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


def make_dataset(f, x0, num_trajectories, num_dim, begin, end, noise):
    xx = [] # states
    projeds = [] # observations

    for i in trange(num_trajectories):
        t, x, projed = gen_data_diffeq(f, random_proj,
                                       t=(0, 12500), x0=x0 + 0.01, dim=num_dim, noise="normal",
                                       ivp_kwargs={'max_step': 0.05},
                                       noise_kwargs={"loc": 0, "scale": noise},)
        t = t[begin:end]
        xx.append(x[begin:end])
        projeds.append(projed[begin:end])

    xx = np.stack(xx, axis=0)
    projeds = np.stack(projeds, axis=0)

    xs = xx
    ys = projeds
    us = np.zeros((xx.shape[0], xx.shape[1], 1))

    filename = f"{f.__name__}_{num_trajectories}trajectories_{num_dim}dim_{begin}to{end}_noise{noise}.npz"
    np.savez(filename, x = xs, y = ys, u = us)


def generate_lorenz():
    """
    * begin/end: ommiting first 500
    * num_trajectory = 1  # one long trajectory
    * noise level: noise = 0.05   OR   0.2
    (0.05 is the value Anne used, 1 is the value that memming's lab used)
    * num_dim = 3 # we will stick with 3 for our comparison purpose!
    1 x 1 x 2 x 1 = 2 example datasets
    """
    for num_trajectory in [1]:
        for (begin, end) in [(500, 20500)]:
            for noise in [0.05, 0.2]:
                for num_dim in [3]:
                    make_dataset(lorenz, x0=np.array([0, 1, 1.05]), num_trajectories=num_trajectory, num_dim=num_dim, begin=begin, end=end, noise=noise)


def generate_vdp():
    """
    * begin/end: ommiting first 500
    * num_trajectory = 1  # one long trajectory
    * noise level: noise = 0.05   OR   0.2
    (0.05 is the value Anne used, 1 is the value that memming's lab used)
    * num_dim = 2 # we will stick with 3 for our comparison purpose!
    1 x 1 x 2 x 1 = 6 example datasets
    """
    for num_trajectory in [1]:
        for (begin, end) in [(500, 20500)]:
            for noise in [0.05, 0.2]:
                for num_dim in [2]:
                    make_dataset(vdp, x0=np.array([0.1, 0.1]), num_trajectories=num_trajectory, num_dim=num_dim, begin=begin, end=end,
                                     noise=noise)


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'lorenz':
        generate_lorenz()
    elif len(sys.argv) == 2 and sys.argv[1] == 'vdp':
        generate_vdp()
    else:
        print(f"usage: python {sys.argv[0]} (lorenz|vdp)", file=sys.stderr)
