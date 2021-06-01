#%%
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter1d

from proSVD import proSVD

#%% dumb streaming svd or window svd for comparison
def get_streamingSVD(X, k, l1, l, num_iters, window=True):
    Us = np.zeros((X.shape[0], k, num_iters))
    Ss = np.zeros((k, num_iters))
    
    U, S, V_T = np.linalg.svd(X[:, :l1], full_matrices=False)
    Us[:, :, 0] = U[:, :k]
    
    t = l1
    if window:
        start = t
    else:
        start = 0
        
    for j in range(1, num_iters):
        U, S, V_T = np.linalg.svd(X[:, start:t+l], full_matrices=False)
        t = t + l
        Us[:, :, j] = U[:, :k]
        Ss[:, j] = S[:k]
        if window:
            start = t
    return Us, Ss
    
# %%
# load data - random session
path = '/hdd/pgupta/lfads-neural-stitching-reproduce/'
file_loc = 'export_v05_broadbandRethreshNonSorted_filtered/subject_Pierre.date_2016-08-19.saveTagGroup_1.saveTag_2,3,5_export.mat'

# using loadmat like this, syntax equals matlab struct (struct.field)
mat_contents = sio.loadmat(path+file_loc, struct_as_record=False, squeeze_me=True, verify_compressed_data_integrity=False)  
data = mat_contents['data']
spikes = data.spikeRasters # 3d np array in (trial, longest_time, neuron)

# %%
# getting spikes aligned, concatenating all trials

align_to = 'GoCue' # 'Move' # 'GoCue' #
num_steps = 1000
num_trials = 16
bin_size = 15 # in ms

reduced_bins = int(num_steps / bin_size)
trial_types = []
    
reach_labels = dict(DownLeft=0, Left=1, UpLeft=2, Up=3, UpRight=4, Right=5, DownRight=6)

spikes = np.zeros((24, num_trials * reduced_bins))
t = 0
for i in range(num_trials):
    trial_types.append(data.targetDirectionName[i])
    
    if align_to == 'TargetOnset':
        start = data.TargetOnset[i]
    elif align_to == 'GoCue':
        start = data.GoCue[i]
    elif align_to == 'Move':
        start = data.Move[i] - 300
        
    end = start + num_steps
    curr_spikes = np.copy(data.spikeRasters[i, start:end, :].T)
    
    # reducing bin size
    spikes_reduced = np.zeros((data.nUnits, reduced_bins))
    for j in range(24):
        t1 = 0
        for k in range(spikes_reduced.shape[1]):
            spikes_reduced[j, k] = np.sum(curr_spikes[j, t1:t1+bin_size])
            t1 += bin_size
    
    # smoothing
#     for j in range(spikes_reduced.shape[0]):
#         spikes_reduced[j, :] = gaussian_filter1d(spikes_reduced[j, :], sigma=5)
    
    spikes[:, t:t+reduced_bins] = spikes_reduced
    t += reduced_bins
    
# smoothing spikes
spikes = gaussian_filter1d(spikes, sigma=2) # smooths by row

# center data (rows because a is neurons x time)
spikes -= spikes.mean(axis=1)[:, np.newaxis]

#%% doing streamingSVD on smoothed spikes, projecting spikes onto the learned subspace

k = 6  # num cols to init with
l = 1    # num cols processed per iter
decay = 1 # 'forgetting' to track nonstationarity. 1 = no forgetting
l1 = k   # number of singular basis vectors to keep 
num_iters = np.ceil((spikes.shape[1] - l1) / l).astype('int') # num iters to go through once

# run streaming svd - returns Qcoll, shape (n, l1, num_iters)
# Qts, Ss, Qs = get_ssSVD(spikes, k, l1, l, num_iters, decay_alpha=decay, trueSVD=True,
#                        silent=True)


A_init = spikes[:, :l1]
pro = proSVD(k, history=spikes.shape[1]-l1, trueSVD=False)
pro.initialize(A_init)
for i in np.arange(l1, l1+num_iters):
    dat = spikes[:, i:i+1]
    pro.updateSVD(dat)
# Qts, Ss, Qs = (pro.Qts, pro.Ss, pro.Qs)
Qs = pro.Qs

#%% doing full SVD
Us, Ss = get_streamingSVD(spikes, k, l1, l, num_iters, window=False)

# %% projecting each timepoint of neural activity onto the subspace learned for that chunk
all_projs_stream = np.zeros((Qs.shape[1], Qs.shape[2]*l))
for i in range(num_iters):
    Q = Qs[:, :, i] # has first k components
    if i == 0:
        curr_neural = spikes[:, :l1]
        all_projs_stream[:, :l1] = Q.T @ curr_neural 
        t = l1
    else: 
        if t + l > Us.shape[2] * l:
            break
        # aligning neural to Q (depends on l1 and l)
        curr_neural = spikes[:, l1+((i-1)*l):l1+(i*l)]
        
        # projecting curr_neural onto curr_Q (our tracked subspace) and on full svd u
        all_projs_stream[:, t:t+l] = Q.T @ curr_neural
        t += l

all_projs_stream_true = np.zeros((Us.shape[1], Us.shape[2]*l))
for i in range(Us.shape[2]):
    Q = Us[:, :, i]
    if i == 0:
        curr_neural = spikes[:, :l1]
        all_projs_stream_true[:, :l1] = Q.T @ curr_neural
        t = l1
    else: 
        if t + l > Us.shape[2] * l:
            break
        # aligning neural to Q (depends on l1 and l)
        curr_neural = spikes[:, l1+((i-1)*l):l1+(i*l)]
        
        # projecting curr_neural onto curr_Q (our tracked subspace) and on full svd u
        all_projs_stream_true[:, t:t+l] = Q.T @ curr_neural
        t += l

num_remove = all_projs_stream.shape[1] - t
num_remove = all_projs_stream_true.shape[1] - t

if num_remove > 0:
    all_projs_stream_true = all_projs_stream_true[:, :-num_remove]
    all_projs_stream = all_projs_stream[:, :-num_remove]

# np.savez('/hdd/pgupta/ssSVD_results.npz', l1=l1, k=k, spikes=spikes, bin_size=bin_size, Us=Us, Qs=Qs)
# %%
