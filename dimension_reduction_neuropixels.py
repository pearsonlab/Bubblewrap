#%%
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.lines import Line2D

from scipy import io as sio
from scipy.ndimage import gaussian_filter1d
from sklearn import random_projection as rp

import proSVD

#%% loading processed matlab data
# one mouse
dataloc = '/hdd/pgupta/stringer2019/neuropixels/ephysLoad_output/'
file = 'WaksmanwithFaces_KS2.mat'
matdict = sio.loadmat(dataloc+file, squeeze_me=True)
spks = matdict['stall']

# truncate so it doesn't take forever
spks = spks[:, :15000]


def reduce_sparseRP(X, comps=100, eps=0.1, transformer=None):
    np.random.seed(42)
    if transformer is None:
        transformer = rp.SparseRandomProjection(n_components=comps, eps=eps)
    X_new = transformer.fit_transform(X)
    return X_new

#%%
# filename = 'vdp_1trajectories_10000dim_500to20500_noise0.05.npz'
# npz = np.load('/hdd/pgupta/' + filename)
# y = npz['y']
# X = np.squeeze(y.T)

X = spks[:, :1000]
rp_dim = 200
svd_dim = 10

chunk_size = 10
num_iters = int(X.shape[1] / chunk_size)

rp_times = []
svd_times = []
transformer = rp.SparseRandomProjection(n_components=rp_dim)
pro = proSVD.proSVD(svd_dim, trueSVD=False, history=0)
pro.initialize(X[:rp_dim, :svd_dim])
X_rp_svd = np.zeros((svd_dim, X.shape[1])) # final reduced
t = 0
for i in range(num_iters):
    dat = X[:, t:t+chunk_size]
    start_rp = time.time()
    X_rp = reduce_sparseRP(dat.T, transformer=transformer).T
    rp_times.append(time.time() - start_rp)

    start_svd = time.time()
    pro.updateSVD(X_rp)
    X_rp_svd[:, t:t+chunk_size] = pro.Q.T @ X_rp
    svd_times.append(time.time() - start_svd)

    t += chunk_size

rp_times = np.array(rp_times) * 1000 / chunk_size
svd_times = np.array(svd_times) * 1000 / chunk_size


#%% running proSVD
%%time

k = 10  
l1 = k      # cols to init
l = 1       # num cols processed per iter (1 datapoint)
decay = 1   # 'forgetting' 

num_iters = np.ceil((spks.shape[1] - l1) / l).astype('int') # num iters to go through once
print(num_iters)

pro_proj = np.zeros((k, spks.shape[1]))
pro = proSVD.proSVD(k, history=0, trueSVD=True)
pro.initialize(spks[:, :l1])
pro_proj[:, :l1] = pro.Q @ spks[:, :l1]

t = l1
for i in range(num_iters):
    start_rp = time.time()
    X_rp = reduce_sparseRP()
    end_rp = time.time()

    start_svd = time.time()
    pro.updateSVD(spks[:, t:t+l])
    pro_proj[:, t:t+chunk_size] = pro.Q @ spks[:, t:t+chunk_size]
    end_svd = time.time()

    t += chunk_size

#%% getting derivs / distance to final
sigma = 10 # gaussian smoothing
derivs_pro = pro.get_derivs()[:, :k]
# derivs_svd = pro.get_derivs(k=comps, trueSVD=True)
derivs_pro = gaussian_filter1d(derivs_pro, sigma=sigma)

a = np.linalg.norm(pro.Qs - pro.Q[:, :, np.newaxis], axis=(0))
b = np.linalg.norm(pro.Qts - pro.Qt[:, :, np.newaxis], axis=(0))
b = gaussian_filter1d(b, sigma=sigma)

#%% plotting derivs/dist to final basis
start, end = int(0 * num_iters), int(1 * num_iters-1)
wallclockrange = np.arange(start, end) * l * 0.0333 # scale to bins (*l) and seconds (*.033)
xrange = np.arange(start, end) / num_iters

fig, ax = plt.subplots(1, 2, figsize=(9,4))
plt.subplots_adjust(wspace=0.3)
ax[0].plot(wallclockrange, derivs_pro[start:end, :])
ax[0].set(xlabel='seconds of data seen', 
          ylabel='derivative of ssSVD basis vectors', 
          title='ssSVD basis vectors stabilize quickly')

ax[1].plot(xrange, a[:, start:end].T, label='ssSVD')
ax[1].plot(xrange, b[:, start:end].T, color='gray', 
           ls='--', alpha=0.3, label='streaming SVD')
ax[1].set(xlabel='fraction of data seen',
          ylabel='distance of basis vectors to their final position',
          title='ssSVD projection (stably?) \n gets closer to final position')

#%% looking at random sections (no trials)
ig, ax = plt.subplots(2, 3, figsize=(15, 8), sharey='row', sharex='row')

cmap = cm.get_cmap('Dark2')
ts = [1000, 5000, 10000]
curr_spikes = spks[:, 600:650]
full_projs = [pro.Qts[:, :, -1].T.dot(curr_spikes), 
             pro.Qs[:, :, -1].T.dot(curr_spikes)]

plane = 0 # plane 0 is right singular plane 1
ind1, ind2 = plane, plane+1

for i, t in enumerate(ts):
    trial_label = t + l1
    for j, bases in enumerate((pro.Qts, pro.Qs)):
        basis = bases[:, :, t]
        curr_proj = basis.T @ curr_spikes
        
        # projection onto currently learned subspace
        if j == 0:
            ax[j, i].plot(curr_proj[ind1, :], curr_proj[-ind2, :], color=cmap(j))
            ax[j, i].scatter(curr_proj[ind1, 0], curr_proj[-ind2, 0], color=cmap(j))
        else:
            ax[j, i].plot(curr_proj[ind1, :], curr_proj[ind2, :], color=cmap(j))
            ax[j, i].scatter(curr_proj[ind1, 0], curr_proj[ind2, 0], color=cmap(j))
        
        # projection onto final subspace
        full_proj = full_projs[j]
        ax[j, i].plot(full_proj[ind1, :], full_proj[ind2, :], c='k', alpha=.5, ls='--')
        ax[j, i].scatter(full_proj[ind1, 0], full_proj[ind2, 0], c='k')
    
    ax[1, i].set(xlabel='ssSVD basis vector 1', ylabel='ssSVD basis vector 2')
    ax[0, i].set(title='{} bins ({} s) seen'.format(trial_label, (trial_label * bin_size)),
                 xlabel='singular vector 1', ylabel='singular vector 2')

custom_lines = [Line2D([0], [0], color=cmap(1)),
                Line2D([0], [0], color='k', ls='--')]
ax[1, 2].legend(custom_lines, ['ssSVD', 'whole data ssSVD'], loc='lower right')

custom_lines = [Line2D([0], [0], color=cmap(0)),
                Line2D([0], [0], color='k', ls='--')]
ax[0, 2].legend(custom_lines, ['streaming SVD', 'whole data SVD'], loc='lower left')

#%% projecting each timepoint of neural activity onto the subspace learned for that chunk
# %%time

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

np.savez('neuropixel_reduced.npz', ssSVD10=proj_stream_ssSVD)
#%% animating projections
st = 0
end = 1000
nframes = end - st
# projed = pro.Q[:, :k].T @ spks[:, st:end]
projed = projs[1]

fig, ax = plt.subplots()
scat = ax.scatter(projed[0, 0], [projed[1, 0]], cmap=cm.rainbow)
def animate(i, nframes):
    scat.set_offsets(projed[:2, :i].T)
    scat.set_array(np.arange(nframes))
    return scat,

anim = animation.FuncAnimation(fig, animate, 
                               frames=nframes, 
                               interval=15,
                               fargs=(nframes,),
                               blit=True)

xmin, ymin = projed.min(axis=1)[:2]
xmax, ymax = projed.max(axis=1)[:2]
ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

# anim.save('stringer_ephys_anim_stream.mp4')

# %%
