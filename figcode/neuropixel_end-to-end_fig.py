#%%
import time
import numpy as np

import matplotlib.pyplot as plt

from sklearn import random_projection as rp
import streamingSVD.proSVD as proSVD
from field.gqds import GQDS

#%% reduction functions
def reduce_sparseRP(X, comps=100, eps=0.1, transformer=None):
    if transformer is None:
        transformer = rp.SparseRandomProjection(n_components=comps, eps=eps)
    X_new = transformer.fit_transform(X)
    return X_new

def reduce_svd_pro(pro, X, k=100):
    return X @ pro.Q[:, :k]

# embed
def embed_data(X, N):
    n, t = X.shape
    A = np.random.uniform(size=(N,n)) # do this differently?
    Q, R = np.linalg.qr(A)
    return Q @ X

#%% generate data
n = 15             # original dim
N = 1000           # embed dim
k = 15             # reduced dim
t = 1000           # timesteps
t1 = t
loc, sig = 0, 1     # gaussian noise for original dim

# M = get_stable_dynamics_mat(n)
# M = np.random.normal(loc=0, scale=.001, size=(n,n))
# M = 0.5 * (M - M.T)
# X_low= generate_stable_LDS(n=n, t=t, M=M, noise_loc=loc, noise_sig=sig)

X_true = np.random.normal(size=(n, t))
X = embed_data(X_true, N)
# plt.plot(X_true.T)

#%% get timing for each step
rp_dim = 200
svd_dim = 10
iters = 20

# rp
transformer = rp.SparseRandomProjection(n_components=rp_dim)
times = []
for i in range(iters):
    start = time.time()
    X_rp = reduce_sparseRP(X.T, transformer=transformer).T
    times.append(time.time()-start)
rp_times_mean = np.array(times).mean() * 1000
rp_times_std = np.array(times).std() * 1000 / np.sqrt(iters-1)


#%%
l = 1 # one vector at a time
# ssSVD
pro = proSVD.proSVD(svd_dim, trueSVD=False, history=0)
pro.initialize(X_rp[:, :k])
times = []
for i in range(iters):
    start = time.time()
    pro.updateSVD(X_rp[:, :l])
    times.append(time.time()-start)
svd_times_mean = np.array(times).mean() * 1000
svd_times_std = np.array(times).std() * 1000 / np.sqrt(iters-1)


#%% bubblewrap
ignore_iters = 5
dt = 0.1
M = 10

d = svd_dim
num_d = 2
N = 1000
lam = 1e-3
nu = 1e-3

step = 1e-3
# sigma_scale = 1e3
B_thresh = -10
n_thresh = 1e-6
eps = 0

t_wait = 1 # breadcrumbing
gq = GQDS(N, num_d, d, step=step, lam=lam, eps=eps, M=M,
          nu=nu, t_wait=t_wait, n_thresh=n_thresh, B_thresh=B_thresh)

for i in np.arange(M):
    gq.observe(X[i+M, :svd_dim])

gq.init_nodes()
print('Nodes initialized')

times = []
for i in np.arange(M):
    gq.observe(X[i+M, :svd_dim])

for i in np.arange(iters+1): # plus 1 to avoid the first one
    # print(i)
    start = time.time()
    gq.observe(X[i+M, :svd_dim])
    gq.em_step()   
    times.append(time.time()-start)  # time observing and updating
bw_times_mean = np.array(times[ignore_iters:]).mean() * 1000
bw_times_std = np.array(times[ignore_iters:]).std() * 1000 / np.sqrt(iters-1)

#%% plot

label = ['N = 1000', 'N = 10000']
bar_labels = ['rp N->n', 'svd n->k', 'bubblewrap']
width = .3
fig, ax = plt.subplots(figsize=(3,5))
means = [rp_times_mean, svd_times_mean, bw_times_mean]
stds = [rp_times_std, svd_times_std, bw_times_std]

currmean1 = means[0]
currstd = stds[0]
ax.bar(i/2, currmean1, width, yerr=currstd, label=bar_labels[0])

# stack svd
currmean2 = means[1]
currstd = means[1]
ax.bar(i/2, currmean2, width, yerr=currstd, bottom=currmean1, 
    label=bar_labels[1])

# stack bubblewrap
currmean = means[2]
currstd = means[2]
ax.bar(i/2, currmean, width, yerr=currstd, bottom=currmean1+currmean2,
    label=bar_labels[2])
ax.set(ylabel='time (ms)', xlabel='N={}, n={}'.format(N, n))

ax.set(xticks=[])
ax.axhline(33, c='k', ls='--') # 30 hz sampling?
ax.legend()
plt.savefig('figcode/end-to-end.pdf', bbox_inches='tight')

# %%
