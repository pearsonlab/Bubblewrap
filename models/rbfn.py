#%%
"""
An implementation of `Interpretable Nonlinear Dynamic Modeling of Neural Trajectories`
Yuan Zhao, Il Memming Park, NIPS 2016

Equations are exact matches to those in the paper.
Generate data from a van der pol oscillator, fit with MSE, and draw vector field.
Takes ~5 ms to run per step on a 4 GHz Coffee Lake CPU.

"""

from functools import partial
from typing import Callable

import jax.numpy as np
from jax import jit
from jax.api import value_and_grad
from jax.interpreters.xla import DeviceArray


#%%
class RBFN:
    def __init__(self, ker: Callable, params, optimizer: tuple[Callable, ...]) -> None:
        assert {"W", "τ", "c", "σ"} <= params.keys()
        assert params["W"].shape == params["c"].shape
        assert params["W"].shape[0] == params["σ"].size
        assert np.all(params["τ"] > 0) and np.all(params["σ"] > 0)

        self.init_params, self.opt_update, self.get_params = optimizer
        self.opt_update = jit(self.opt_update)
        self.opt_state = self.init_params(params)

        self.ker = ker
        self._mse_vgrad = jit(value_and_grad(self._mse, argnums=2), static_argnums=0)
        self.i = 0

    @property
    def params(self):
        return self.get_params(self.opt_state)

    def g(self, x):
        return self._g(self.ker, x, **self.params)

    def step(self, x, loop=3):
        for _ in range(loop):
            value, grads = self._mse_vgrad(self.ker, x, self.params)
            self.opt_state = self.opt_update(self.i, grads, self.opt_state)
        self.i += 1
        return value

    @staticmethod
    @partial(jit, static_argnums=0)
    def _g(kern, x, W, τ, c, σ):
        return kern(x, c, σ) @ W - np.exp(-(τ ** 2)) * x  # (4)

    @staticmethod
    def _mse(kern: Callable, x: DeviceArray, p: dict[str, DeviceArray]):
        return np.mean(np.square(RBFN._g(kern, x[:-1], p["W"], p["τ"], p["c"], p["σ"]) + x[:-1] - x[1:]))

# %%

# %%
