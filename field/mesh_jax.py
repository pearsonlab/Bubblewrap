import jax.numpy as np
from jax.api import jit


# @jit
def predict_vec(coords_curr, vec_curr, coords_obs, p=1):  
    dists = np.linalg.norm(coords_curr - coords_obs, axis=1)
    weights = 1 / (dists**p)
    return weights @ vec_curr / np.sum(weights)  # Σwᵢvᵢ / Σwᵢ


# @jit
def prediction_loss(coords_curr, vec_curr, coords_obs, vec_obs):
    # Already indexed to bounding points.
    pred = predict_vec(coords_curr, vec_curr, coords_obs)
    return np.sum(np.abs(pred - vec_obs)**2)  # Σ |pred - obs|²


@jit
def spring_energy(coords, neighbors, k=1., l0=1.):
    arr = coords[neighbors]  # (points × n_neighbors × dim)
    mask = (neighbors > -1).astype(np.float32)[..., np.newaxis]

    centered = (arr - coords[:, np.newaxis, :]) * mask + 1e-10 # Zero out mask and prevent sqrt(-0).
    ℓ = np.linalg.norm(centered, axis=2)

    return 0.5 * k * np.sum((ℓ - l0)**2) # energy
