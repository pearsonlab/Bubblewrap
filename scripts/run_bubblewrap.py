import time
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pylab as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D

from bubblewrap import Bubblewrap

from math import atan2, floor

## Load data from datagen/datagen.py 
s = np.load('vdp_1trajectories_2dim_500to20500_noise0.05.npz')
data = s['y'][0]

T = data.shape[0]       # should be 20k
d = data.shape[1]       # should be 2

## Parameters
N = 100             # number of nodes to tile with
lam = 1e-3          # lambda 
nu = 1e-3           # nu
eps = 1e-3          # epsilon sets data forgetting
step = 8e-2         # for adam gradients
M = 30              # small set of data seen for initialization
B_thresh = -10      # threshold for when to teleport (log scale)    
batch = False       # run in batch mode 
batch_size = 1      # batch mode size; if not batch is 1
go_fast = False     # flag to skip computing priors, predictions, and entropy for optimal speed

bw = Bubblewrap(N, d, step=step, lam=lam, M=M, eps=eps, nu=nu, B_thresh=B_thresh, batch=batch, batch_size=batch_size, go_fast=go_fast) 

## Set up for online run through dataset
init = -M
end = T-M
step = batch_size

## Initialize things
for i in np.arange(0, M, step): 
    if batch:
        bw.observe(data[i:i+step]) 
    else:
        bw.observe(data[i])
bw.init_nodes()
print('Nodes initialized')

## Run online, 1 data or batch at a time
for i in np.arange(init, end, step):
    bw.observe(data[i+M:i+M+step])
    bw.e_step()  
    bw.grad_Q()
print('Done fitting all data online')

## Plotting
plt.figure()
plt.plot(bw.pred)
var_tmp = np.convolve(bw.pred, np.ones(500)/500, mode='valid')
plt.plot(var_tmp, 'k')
plt.show()

## Saving data example for later plotting
saving = False
if saving:
    A = np.save('saved_A.npy', bw.A)
    mu = np.save('saved_mu.npy', bw.mu)
    L = np.save('saved_L.npy', bw.L)
    n_obs = np.save('saved_n_obs.npy', bw.n_obs)
    pred = np.save('saved_pred.npy', bw.pred)
    entropy = np.save('saved_entropy.npy', bw.entropy_list)