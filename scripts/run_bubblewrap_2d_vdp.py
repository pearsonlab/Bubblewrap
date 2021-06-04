import time
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pylab as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D

from scipy.integrate import solve_ivp
from scipy.stats import multivariate_normal as mvn

from field.gqds import GQDS
from datagen.diffeq import vanderpol, lorenz
from field.utils import center_mass
from datagen import plots

from math import atan2, floor

## Load data from datagen/datagen.py vdp
s = np.load('vdp_1trajectories_2dim_500to20500_noise0.05.npz')
data = s['y'][0]

T = data.shape[0]       # should be 20k
d = data.shape[1]       # should be 2

## Bubblewrap parameters

N = 100 
lam = 1e-3
nu = 1e-3 
eps = 1e-3

step = 8e-2

M = 30

t_wait = 1 
B_thresh = -10 
n_thresh = 5e-4

batch = False
batch_size = 1 #50

gq = GQDS(N, d, go_fast=False, step=step, lam=lam, M=M, eps=eps, nu=nu, t_wait=t_wait, B_thresh=B_thresh, n_thresh=n_thresh, batch=batch, batch_size=batch_size) #, mu_diff=mu_diff)

## initialize things
for i in np.arange(0,M): #,batch_size):
    gq.observe(data[i]) #:i+batch_size])

gq.init_nodes()
print('Nodes initialized')

## run online
timer = time.time()
times_em = []
times_Q = []
times_obs = []

init = -M
end = T-M
step = batch_size

for i in np.arange(init, end, step):
    # print(i)
    t1 = time.time()
    gq.observe(data[i+M:i+M+step])
    times_obs.append((time.time()-t1)/step)
    t2 = time.time()
    gq.em_step()  
    times_em.append((time.time()-t2)/step) 
    t3 = time.time() 
    gq.grad_Q()
    times_Q.append((time.time()-t3)/step)    

print('Done fitting all data online')
length = floor(len(times_em)/4)
emtime = np.mean(np.array(times_em)[length:])
qtime = np.mean(np.array(times_Q)[length:])
obstime = np.mean(np.array(times_obs)[length:])
print('Average em time: ', emtime)
print('Average Q time: ', qtime)
print('Average observation time: ', obstime)
total = np.array(times_em)+np.array(times_Q)+np.array(times_obs)
print('-------- Total time: ', np.mean(total[length:])*1e3, ' +- ', np.std(total[length:])*1e3)
# print('Average prediction time: ', np.mean(np.array(gq.time_pred)[20:]))


if make_movie:
    writer.finish()

## plotting

plt.figure()
plt.plot(np.array(gq.pred))
var_tmp = np.convolve(np.array(gq.pred), np.ones(500)/500, mode='valid')
plt.plot(var_tmp, 'k')

plt.figure()
plt.plot(np.array(gq.entropy_list))
plt.hlines(np.log2(N), 0, T, 'k', '--')
var_tmp = np.convolve(np.array(gq.entropy_list), np.ones(500)/500, mode='valid')
plt.plot(var_tmp, 'k')

plt.figure()
axs = plt.gca()
axs.plot(data[:i+1+M+step,0], data[:i+1+M+step,1], color='gray', alpha=0.8)
for n in np.arange(N):
    el = np.linalg.inv(gq.L[n][:2,:2])
    sig = el.T @ el
    u,s,v = np.linalg.svd(sig)
    width, height = np.sqrt(s[0]*9), np.sqrt(s[1]*9) #*=4
    angle = atan2(v[0,1],v[0,0])*360 / (2*np.pi)

    el = Ellipse((gq.mu[n,0],gq.mu[n,1]), width, height, angle, zorder=8)
    el.set_alpha(0.2)
    el.set_clip_box(axs.bbox)
    el.set_facecolor('r') 
    axs.add_artist(el)

mask = np.ones(gq.mu.shape[0], dtype=bool)
if gq.dead_nodes:
    mask[np.array(gq.dead_nodes)] = False
axs.scatter(gq.mu[mask,0], gq.mu[mask,1], c='k' , zorder=10)


plt.figure()
plt.plot(times_em[M:])
plt.plot(times_Q[M:])
plt.plot(times_obs[M:])

plt.plot(gq.time_pred[M:])

plt.show()