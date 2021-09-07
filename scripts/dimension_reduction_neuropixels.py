#%%
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname("__file__")))

import time
from pathlib import Path

import numpy as np
import proSVD
from scipy import io as sio
from sklearn import random_projection as rp

#%% loading processed matlab data
# one mouse
assert len(sys.argv) == 2
file = Path(sys.argv[1])

matdict = sio.loadmat(file, squeeze_me=True)
spks = matdict['stall']

# truncate so it doesn't take forever
spks = spks[:, :15000]

def reduce_sparseRP(X, comps=100, eps=0.1, transformer=None):
    np.random.seed(42)
    if transformer is None:
        transformer = rp.SparseRandomProjection(n_components=comps, eps=eps)
    X_new = transformer.fit_transform(X)
    return X_new

#%% parameters
# reducing ~2000 neuropixels channels to 200 dims through random projections
# then reducing that to 10 dims through ssSVD
X = spks
rp_dim = 200        # random projections
svd_dim = 10        # ssSVD
l = chunk_size = 10 # num cols to process per iter
k = 10              # num components to keep
l1 = k              # cols to init
decay = 1           # 'forgetting' 
num_iters = np.ceil((spks.shape[1] - l1) / l).astype('int') # num iters to go through once

#%% running random projections + ssSVD
pro_proj = np.zeros((k, spks.shape[1]))
pro = proSVD.proSVD(k, history=num_iters, trueSVD=True)
pro.initialize(spks[:, :l1])
pro_proj[:, :l1] = pro.Q.T @ spks[:, :l1]

t = l1
transformer = rp.SparseRandomProjection(n_components=rp_dim)
for i in range(num_iters):
    if i & 15 == 0:  # Check if multiple of 16
        print(f"ssSVD Iteration {i}/{num_iters}")
    start_rp = time.time()
    X_rp = reduce_sparseRP(X, transformer=transformer)
    end_rp = time.time()

    start_svd = time.time()
    pro.updateSVD(spks[:, t:t+l])
    end_svd = time.time()

    t += chunk_size

#%% projecting each timepoint of neural activity onto the subspace learned for that chunk
proj_stream_ssSVD = np.zeros((pro.Qs.shape[1], pro.Qs.shape[2]*l))
proj_stream_SVD = np.zeros((pro.Qts.shape[1], pro.Qts.shape[2]*l))
projs = [proj_stream_ssSVD, proj_stream_SVD]
bases = [pro.Qs, pro.Qts]

for i, basis in enumerate(bases):
    curr_basis = basis[:, :, i] # has first k components
    t = 0
    for j in range(pro.Qs.shape[2]-2): # -1 to deal with bin issues
        if j == 0: # init projection
            curr_neural = spks[:, :l1]
            projs[i][:, :l1] = curr_basis.T @ curr_neural
            t = l1
        else: 
            # aligning neural to Q (depends on l1 and l)
            curr_neural = spks[:, l1+t:l1+t+l]
            # projecting curr_neural onto curr_Q (our tracked subspace) and on full svd u
            projs[i][:, t:t+l] = curr_basis.T @ curr_neural
            t += l

np.savez(file.parent, ssSVD10=proj_stream_ssSVD)

# %%
