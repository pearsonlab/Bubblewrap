#%%
from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as np
import networkx as nx
import numpy as onp
import seaborn as sns
from jax import jit
from jax.api import value_and_grad
from jax.interpreters.xla import DeviceArray
from models.rbfn import RBFN


def gen_grid(m=4, n=4):
    G = nx.grid_2d_graph(m, n)
    points = np.array(G.nodes)
    keys = {x: i for i, x in enumerate(list(G.nodes))}

    neighbors = -1 * onp.ones((points.shape[0], 4), dtype=int)
    n_neighbor = -1 * onp.ones(points.shape[0])
    for i, point in enumerate(G.nodes):
        ed = [keys[edge[1]] for edge in G.edges(point)]
        neighbors[i, : len(ed)] = ed
    return points, np.array(neighbors), np.array(n_neighbor)


class RBFNSpring(RBFN):
    def __init__(self, ker: Callable, params, optimizer: tuple[Callable, ...],
        *, params_spr: dict[str, float], nb: DeviceArray
    ):
        super().__init__(ker, params, optimizer)
        assert nb.shape[0] == self.params["σ"].shape[0]
        self.nb = nb
        
        assert {"k", "l0"} <= params_spr.keys()
        self.p_spr = params_spr

        self._obj = partial(
            jit(value_and_grad(self._mse_spring, argnums=2), static_argnums=0),
            k=self.p_spr["k"],
            l0=self.p_spr["l0"],
        )

    @staticmethod
    def _mse_spring(kern: Callable, x: DeviceArray, p: dict[str, DeviceArray], mask: Optional[DeviceArray], 
                    nb: DeviceArray, **kwargs):
        return RBFNSpring._mse(kern, x, p, mask) + RBFNSpring.spring_energy(p["c"], nb, **kwargs)

    @staticmethod
    def spring_energy(coords, neighbors, k=1.0, l0=1.0):
        arr = coords[neighbors]  # (points × n_neighbors × dim)
        mask = (neighbors > -1).astype(np.float32)[..., np.newaxis]

        centered = (arr - coords[:, np.newaxis, :]) * mask + 1e-7  # Zero out mask and prevent sqrt(-0).
        ℓ = np.linalg.norm(centered, axis=2)
        return 0.5 * k * np.sum((ℓ - l0) ** 2)  # energy
