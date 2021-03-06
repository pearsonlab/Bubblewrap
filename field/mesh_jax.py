import jax.numpy as np
from jax.api import jit


@jit
def predict_vec(coords_curr, vec_curr, coords_obs, p=1):  
    dists = np.linalg.norm(coords_curr - coords_obs, axis=1)
    # print("test", dists)
    weights = 1 / (dists**p)
    return weights @ vec_curr / np.sum(weights)  # Σwᵢvᵢ / Σwᵢ

@jit
def loss(coords_curr, vec_curr, coords_obs, vec_obs):
    # Already indexed to bounding points.
    pred = predict_vec(coords_curr, vec_curr, coords_obs)
    return np.sum(np.abs(pred - vec_obs)**2)
