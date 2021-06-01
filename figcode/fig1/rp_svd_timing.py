#%%
import time
import numpy as np
np.set_printoptions(precision=4)
# import cupy as cp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn import random_projection as rp
from scipy.spatial.distance import pdist

import proSVD

#%% reduction functions
def reduce_sparseRP(X, comps=100, eps=0.1, transformer=None):
    if transformer is None:
        transformer = rp.SparseRandomProjection(n_components=comps, eps=eps)
    X_new = transformer.fit_transform(X)
    return X_new

def reduce_svd(X, k=100):
    u, s, v = np.linalg.svd(X, full_matrices=False)
    return X.T @ u[:, :k]

# embedding function
def embed_data(X, N):
    n, t = X.shape
    A = np.random.uniform(size=(N,n)) # do this differently?
    Q, R = np.linalg.qr(A)
    return Q @ X

# %% RP accuracy functions

def get_dist_list(X, k=1):
    # print(X.shape) 
    dists = pdist(X.T, 'sqeuclidean')
    # dists /= dists.max()
    # dists = euclidean_distances(X.T, squared=True).ravel()
    # nonzero = dists != 0
    # dists = dists[nonzero]
    return dists

# 
def get_distortion(X, Y, form='max_dist'):
    if form == 'max_dist':
        dists1 = get_dist_list(X[:, :100])
        dists2 = get_dist_list(Y[:, :100])
        ret = get_pairwise_error(dists1, dists2)
        # print(dists1.shape, dists2.shape)
        # ret = np.mean(np.abs(dists1 - dists2))
        # ret = np.linalg.norm(dists1 - dists2) / dists1.shape[0]
    elif form == 'traut':
        lens1 = np.linalg.norm(X, axis=0)
        lens2 = np.linalg.norm(Y, axis=0)
        ret = get_distortion_traut(lens1, lens2)
    else:
        print('yo')
    return ret

# eps = abs( lambda * norm(e) - norm(r) ) / norm(r)
# finds max eps for all pairs given norms
def _get_distortion_traut(lengths1, lengths2):
    diffs = lengths1[:, np.newaxis] - lengths2[np.newaxis, :]
    diffs = np.abs(diffs)
    diffs = diffs / lengths1[:, np.newaxis]
    diffs[~np.isfinite(diffs)] = 0
    ret = np.max(diffs)
    return ret

# assume pairwise dists are in an nx1 array
def get_pairwise_error(dists1, dists2):
    diffs = np.abs((dists2 / dists1) - 1)
    return np.percentile(diffs, 95)

# gets distortion as from Trautmann 2019
# "The distortion of a set of manifolds is computed by 
# finding the maximum distortion of the vectors between 
# all pairs of points on anyof the manifolds, including 
# those between different manifolds."
def get_distortion_traut(lengths1, lengths2):
    # a = _get_distortion_traut(lengths1, lengths1)
    # b = _get_distortion_traut(lengths2, lengths2)
    c = _get_distortion_traut(lengths1, lengths2)
    # d = _get_distortion_traut(lengths2, lengths1)
    ret = np.max([d])
    return ret


def get_accuracy_rp(X, dists, rp_range, iters=20): # given n channels, loops through chunk size

    accs = np.zeros((rp_range.shape[0], 2))  # 2 for rp and svd
    times = np.zeros((rp_range.shape[0], 2))
    for i, rp_dim in enumerate(rp_range):
        start_svd = time.time()
        X_svd = reduce_svd(X, k=rp_dim)
        times[i, 1] = time.time() - start_svd

        avg_dist_svd = get_distortion(X, X_svd, form='traut')

        avg_dists_rp = np.zeros((iters))
        avg_times = np.zeros((iters)) # TODO: timing in this function?
        for j in range(iters):
            transformer = rp.SparseRandomProjection(n_components=rp_dim)

            start_rp = time.time()
            X_rp = reduce_sparseRP(X.T, transformer=transformer).T
            avg_times[j] = time.time() - start_rp
            avg_dists_rp[j] = get_distortion(X, X_rp, form='traut')

        times[i, 0] = avg_times.mean()
        accs[i, 0] = avg_dists_rp.mean()
        accs[i, 1] = avg_dist_svd.mean()
    return accs, times

def get_accuracy_6_dim(X, dists, rp_range, k=6, form='traut', iters=20):
    accs = np.zeros((rp_range.shape[0], 2)) # mean on first, sem on 2nd
    times = np.zeros(accs.shape)
    for i, rp_dim in enumerate(rp_range):
        avg_dists = np.zeros((iters))
        avg_stds = np.zeros((iters))
        avg_times = np.zeros((iters)) 
        transformer = rp.SparseRandomProjection(n_components=rp_dim)
        iterstime = time.time()
        pro = proSVD.proSVD(k, trueSVD=False)
        pro.initialize(X[:rp_dim, :k])
        # breakpoint()
        for j in range(iters):
            start_time = time.time()
            X_rp = reduce_sparseRP(X.T, transformer=transformer).T
            u, s, v = np.linalg.svd(X_rp, full_matrices=False)
            # pro._updateSVD(X_rp)
            # X_rp_svd = pro.Q.T @ X_rp
            X_rp_svd = u[:, :k].T @ X_rp
            avg_times[j] = time.time() - start_time
            avg_dists[j] = get_distortion(X, X_rp_svd, form=form)
        iterstime = time.time() - iterstime
        print('done \t {} rp_dim, \t {:.2f}'.format(rp_dim, iterstime))
        accs[i, 0] = avg_dists.mean()
        accs[i, 1] = avg_dists.std()
        times[i, 0] = avg_times.mean()
        times[i, 1] = np.std(avg_times, ddof=1)
    return accs, times

#%% Generate data
n = 15             # original dim
N = 10000           # embed dim
k = 10               # reduced dim
t = 10            # timesteps
t1 = t
loc, sig = 0, 1     # gaussian noise for original dim
form = 'max_dist'

# M = get_stable_dynamics_mat(n)
# M = np.random.normal(loc=0, scale=.001, size=(n,n))
# M = 0.5 * (M - M.T)
# X_low= generate_stable_LDS(n=n, t=t, M=M, noise_loc=loc, noise_sig=sig)
X_low = np.random.normal(size=(n, t))
X_embed = embed_data(X_low, N)
# X_embed = np.random.normal(size=(N, t))
X = X_embed[:, :t1]
plt.plot(X_low.T)

#%% running distortion/timing
iters = 2           # iters for each of RP, ssSVD
# rp_range = np.array([10, 50, 100, 500, 1000, 5000])
# rp_range = np.ceil(np.logspace(1, 4, 30)).astype('int')
rp_range = np.arange(10, 10000, 1000) # range of RP dims to run
# rp_range = np.append(rp_range, [7500, 1e4]).astype('int')


dists = get_dist_list(X[:, :100], k=2) # only 100 points for memory
accs, times = get_accuracy_6_dim(X, dists, rp_range, k=k, 
                                 form=form, iters=iters)

# getting just svd N -> k
start = time.time()
u, s, v = np.linalg.svd(X)
X_svd = u[:, :k].T @ X
baseline_svd_time = time.time() - start
baseline_svd = get_distortion(X, X_svd, form=form)

# getting just rp N -> k
transformer = rp.SparseRandomProjection(n_components=k)
baseline_rps = np.zeros((iters))
baseline_stds = np.zeros((iters))
baseline_times = np.zeros((iters))
for i in range(iters):
    start = time.time()
    X_rp = reduce_sparseRP(X.T, transformer=transformer).T
    baseline_times[i] = time.time() - start
    baseline_rps[i] = get_distortion(X, X_rp, form=form)
baseline_rp = baseline_rps.mean()
baseline_std = baseline_stds.mean()
baseline_rp_time = baseline_times.mean()

# saving for figure
# np.savez('/hdd/pgupta/rp_dist_timing_persample.npz', rp_range=rp_range, 
#          accs=accs, times=times, baseline_svd=baseline_svd, 
#          baseline_rp=baseline_rp, baseline_svd_time=baseline_svd_time,
#          baseline_rp_time=baseline_rp_time, t=t)


#%% plots
stop = rp_range[-1]
cmap = cm.get_cmap('cool')
c0 = cmap(0.9)
c1 = cmap(0)

labels = ['RP $N$ to $n$, \n SVD $n$ to $k$', 
          'SVD $N$ to $k$', 
          'RP $N$ to $k$']

fig, axs = plt.subplots(1, 3, figsize=(12,3))
fig.subplots_adjust(wspace=0.3, hspace=0.5)

# distortion
g = axs[0].errorbar(rp_range[:stop], accs[:stop, 0], accs[:stop, 1],
                label=labels[0], c='k', ecolor='gray')
axs[0].axhline(baseline_svd, color=c0, ls='--', label=labels[1])
axs[0].axhline(baseline_rp, color=c1, ls='--', label=labels[2])
axs[0].set(xlabel='$n$', #ylim=(0, 6),
       ylabel='distortion (ϵ)',
       title='distortion of reducing \n $N=${:.0e} dims to $k={}$ dims'.format(N, k))
axs[0].legend()

# timing
h = axs[1].errorbar(rp_range[1:stop], times[1:stop, 0], times[1:stop, 1], 
                 label=labels[0], c='k', ecolor='gray')
axs[1].axhline(baseline_svd_time, color=c0, ls='--', label=labels[1])
axs[1].axhline(baseline_rp_time, color=c1, ls='--', label=labels[2])
axs[1].set(xlabel='$n$', ylabel='time (ms)', 
        title='time to reduce \n $N=${:.0e} dims to $k={}$ dims'.format(N, k))
axs[1].legend()

# both
im = axs[2].scatter(times[:, 0], accs[:, 0], c=rp_range, 
                    cmap=cmap, norm=mcolors.LogNorm())
axs[2].scatter(baseline_rp_time, baseline_rp, color=c1, edgecolors='k', linewidth=1.5)
axs[2].scatter(baseline_svd_time, baseline_svd, color=c0, edgecolors='k', linewidth=1.5)
# a = np.where(rp_range == 924)[0][0]
# axs[2].scatter(times[a, 0], accs[a, 0], color=cmap(19/30), edgecolors='k', linewidth=1.5)
axs[2].set(xlabel='time (ms)', ylabel='distortion (ϵ)',
        title='pareto front of \n distortion vs timing')
cbar = fig.colorbar(im, ax=axs[2])
cbar.set_label('$n$', rotation=270)

# plt.savefig('figures/svd_rp.svg', bbox_inches='tight')
# %%

# %%
