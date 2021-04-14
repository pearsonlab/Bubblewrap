import jax.numpy as np
from jax.api import jit


# @jit
def predict_vec(coords_curr, vec_curr, coords_obs, p=1): 
    # summed = 0
    decay = 1
    # for _,h in enumerate(coords_obs):
    dists = np.linalg.norm(coords_curr - coords_obs[-1], axis=1)
    weights = 1 / (dists**p)
    summed = (weights @ vec_curr / np.sum(weights))/decay  # Σwᵢvᵢ / Σwᵢ
    dists2 = np.linalg.norm(coords_curr - coords_obs[-2], axis=1)
    weights2 = 1 / (dists2**p)
    summed2 = summed + (weights2 @ vec_curr / np.sum(weights2))/2
        # decay *= 1.05   
    return summed2


# @jit
def prediction_loss(coords_curr, vec_curr, coords_obs, vec_obs):
    # Already indexed to bounding points.
    pred = predict_vec(coords_curr, vec_curr, coords_obs)
    return np.sum(np.abs(pred - vec_obs)**2)   # Σ |pred - obs|²

@jit
def spring_energy(coords, neighbors, d, vectors, k=1.):
    arr = coords[neighbors]  # (points × n_neighbors × dim)
    mask = (neighbors > -1).astype(np.float32)[..., np.newaxis]

    centered = (arr - coords[:, np.newaxis, :]) * mask + 1e-10 # Zero out mask and prevent sqrt(-0).
    # mask2 = np.squeeze(d)*(1 - np.squeeze(mask))
    ℓ = np.squeeze(np.linalg.norm(centered, axis=2)) #+ mask2

    l1 = 1/ℓ
    v = vectors[neighbors]
    dotprod = 0
    for i,w in enumerate(v):
        dotprod += ((vectors[i]@w.T) * l1[i]) @ mask[i]

    # ℓ[np.abs(ℓ)<1e-8] = d

    return 0.5 * k * np.sum((ℓ - d)**2) + 1000*np.sum(dotprod)/coords.shape[0] # energy


def total_loss(coords, neighbors, vec, coords_obs, vec_obs, d, k=1.):
    # summed = 0 
    # for _,i in enumerate(coords):
    # mask = (neighbors > -1).astype(np.float32)[..., np.newaxis]
    # summed = np.squeeze(np.sum((vec_obs@(coords[:, np.newaxis, :])*mask)))
    return prediction_loss(coords, vec, coords_obs, vec_obs) + 0.02*spring_energy(coords, neighbors, d, vec) #+ summed/(2*coords.shape[0]) 
