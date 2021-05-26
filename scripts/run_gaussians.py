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


# s = np.load('jpca_reduced.npy')
# data = s.reshape((s.shape[0]*s.shape[1], s.shape[2]))
# data = np.load('proj_cam1.npy').T
# T = data.shape[0]
# d = data.shape[1]

## parameters
## TODO: change def since not meshgrid-ing?

num_d = 3
d = 2
N = 8 #num_d**d
step = 9e-1 #8e-2
lam = 1e-3
nu = 1e-3  #2
# sigma_scale = 1e3 #1e4
# mu_scale = 2
eps = 1e-2

M = 30
T = 5000    #500 + M
P = 0 

t_wait = 1 #M #100
B_thresh = -10 #5.8
n_thresh = 5e-4

batch = False
batch_size = 1 #50

gq = GQDS(N, num_d, d, step=step, lam=lam, M=M, eps=eps, nu=nu, t_wait=t_wait, B_thresh=B_thresh, n_thresh=n_thresh, batch=batch, batch_size=batch_size)

## Get data from vdp dataset
# s = np.load('vdp_1trajectories_2dim_500to20500_noise0.2.npz')
# s = np.load('lorenz_1trajectories_3dim_500to20500_noise0.2.npz')
# data = s['x'][0]    # or grab 2 trajectories as example

## generate data from 8 gaussians, transitioning deterministically
np.random.seed(42)
angles = np.arange(0,2*np.pi, 2*np.pi/8)
xs = 15*np.cos(angles)
ys = 15*np.sin(angles)
mus = np.stack((xs, ys)).T
sig = np.eye(d)
data = np.zeros((T, d))
ind = 0
data_mu = [None]*8
for t in np.arange(0,T):
    data[t,:] = mvn(mean=mus[ind%8], cov=sig).rvs()
    if not np.all(data_mu[ind%8]): 
        data_mu[ind%8] = [data[t,:]]
    else:
        data_mu[ind%8].append(data[t,:]) 
    if t%2 == 0: ind += 1

data_covs = []
data_means = []
for n in np.arange(8):
    data_covs.append(np.cov(np.array(data_mu[n]).T))
    data_means.append(np.mean(data_mu[n], axis=0))

# breakpoint()

def BC_dist(mean1, mean2, cov1, cov2):
    cov = (1 / 2) * (cov1 + cov2)

    T1 = (1 / 8) * (
        np.sqrt((mean1 - mean2) @ np.linalg.inv(cov) @ (mean1 - mean2).T)
    )
    T2 = (1 / 2) * np.log(
        np.linalg.det(cov) / np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2))
    )

    return T1 + T2

# # breakpoint()
# data = np.vstack((s['y'][0], s['y'][1]))
# x0, y0 = (0.1, 0.1)
# dt = 0.1
# sln = solve_ivp(vanderpol, (0, T), (x0, y0), args=(), dense_output=True, rtol=1e-6)
# t = np.linspace(0, dt*T, T)
# data = sln.sol(t).T * 10
# np.random.seed(42)
# data += np.random.randn(data.shape[0],data.shape[1])*0.05*np.std(data)

## initialize things
for i in np.arange(0,M): #,batch_size):
    gq.observe(data[i]) #:i+batch_size])
    # print(np.cov(data[:i+1].T, bias=True))
    # breakpoint()


gq.init_nodes()
print('Nodes initialized')

# Visualization
make_movie = False

if make_movie:
    ## Plotting during mesh refinement
    # fig, axs = plt.subplots(ncols=2, figsize=(6, 3), dpi=100)
    fig = plt.figure()
    # axs = plt.gca(projection='3d')
    axs = plt.gca()
    # axs.view_init(40,23)
    # parameters for animation
    sweep_duration = 5#15
    hold_duration = 10
    total_duration = sweep_duration + hold_duration
    fps = 5#15

    # setup animation writer
    import matplotlib.animation
    writer_class = matplotlib.animation.writers['ffmpeg']
    writer = writer_class(fps=fps, bitrate=1000)
    writer.setup(fig, 'bubblewrap_2d_gaussians.mp4')

## run online
timer = time.time()
times = []
times_obs = []

init = -M
end = T-M
step = batch_size

dists = [None]*8

for i in np.arange(init, end, step):
    # print(i)
    t1 = time.time()
    gq.observe(data[i+M])
    times_obs.append(time.time()-t1)
    t2 = time.time()
    gq.em_step()    
    times.append(time.time()-t2)    

    if make_movie:
        if True: #i < 200 or i > 300:
            # plots.plot3d_color(data[:i+1+M], t[:i+1+M], axs, alpha=1) #, cmap='PuBu')
            axs.scatter(data[:i+1+M+step,0], data[:i+1+M+step,1], color='m')
            axs.plot(data[i+M+step-1:i+1+M+step,0], data[i+M+step-1:i+1+M+step,1], lw=2, color='b')
            for n in np.arange(N):
                if n in gq.dead_nodes: # or (gq.n_obs[n] < 0.08):
            # #         ## don't plot dead nodes
                    pass
                else:
                    el = np.linalg.inv(gq.L[n][:2,:2])
                    sig = el.T @ el
                    u,s,v = np.linalg.svd(sig)
                    width, height = s[0]*4, s[1]*4 #*=4
                    # if width>1e5 or height>1e5:
                    #     pass
                    # else:
                    angle = atan2(v[0,1],v[0,0])*360 / (2*np.pi)
                    # breakpoint()
                    el = Ellipse((gq.mu[n,0],gq.mu[n,1]), width, height, angle, zorder=8)
                    el.set_alpha(0.3)
                    el.set_clip_box(axs.bbox)
                    el.set_facecolor('r')  ##ed6713')
                    axs.add_artist(el)
                
                    axs.text(gq.mu[n,0]+0.1, gq.mu[n,1], s=str(n))

                    targ = np.argmin(np.linalg.norm(gq.mu[n]-np.array(data_means), axis=1))
                    dist = BC_dist(gq.mu[n], data_means[targ], sig, data_covs[targ])
                    if not np.all(dists[n]): dists[n] = [dist]
                    else: dists[n].append(dist)

        # else: #i between 200 and 300
        #     # find node closest to data point
        #     node = np.argmax(gq.alpha)
        #     A = gq.A.copy()
        #     A[A<=(1/N)] = 0
        #     plots.plot_color(data[:i+M+1, 0], data[:i+M+1, 1], t[:i+M+1], axs, alpha=1) #, cmap='PuBu')
        #     for j in np.arange(N):
        #         if A[node,j] > 0 and not node==j:
        #             print('Arrow from ', str(node), ' to ', str(j))
        #             axs.arrow(gq.mu[node,0], gq.mu[node,1], gq.mu[j,0]-gq.mu[node,0], gq.mu[j,1]-gq.mu[node,1], length_includes_head=True, width=A[node,j], head_width=0.8, color='k', zorder=9)


            # plt.xlim([0,1600])
            # plt.ylim([-25,15])
            # axs.set_zlim([-1550,0])

            mask = np.ones(gq.mu.shape[0], dtype=bool)
            if gq.dead_nodes:
                mask[np.array(gq.dead_nodes)] = False
            # mask[gq.n_obs<1e-8] = False
            # breakpoint()
            axs.scatter(gq.mu[mask,0], gq.mu[mask,1], c='k' , zorder=10)

            # axs.set_xticks([-15,15])
            # axs.set_yticks([-15,20])
            axs.set_xticks([])
            axs.set_yticks([])

            plt.draw()
            writer.grab_frame()

            # if i >= 200 and i <= 300:
            #     writer.grab_frame()

            # if i+M in [M-1, floor(T/4)-1, floor(T/2)-1, T-1]:
            #     figS = plt.gcf()
            #     figS.savefig('vdp_'+str(i+1)+'.svg', bbox_inches='tight')
            
            # axs[0].cla()
            # axs[1].cla()
            axs.cla()


    # print(i)
    if (i+M) % 50 == 0  and i>0:
        print(i+M, 'frames processed. Time elapsed ', time.time()-timer)

print('Done fitting all data online')
print('Average cycle time: ', np.mean(np.array(times)[20:]))
print('Average observation time: ', np.mean(np.array(times_obs)[20:]))
# print('Average prediction time: ', np.mean(np.array(gq.time_pred)[20:]))

# breakpoint()

if make_movie:
    writer.finish()


fig, axs = plt.subplots(2, 8)
# vmin = -1 #np.min(data_covs)
# vmax = 6 #np.max(data_covs)
for n in np.arange(N):
    el = np.linalg.inv(gq.L[n][:2,:2])
    sig = el.T @ el
    # print(n, sig)
    # breakpoint()
    targ = np.argmin(np.linalg.norm(gq.mu[n]-np.array(data_means), axis=1))
    dist = BC_dist(gq.mu[n], data_means[targ], sig, data_covs[targ])
    print('Final dist for ', n, ': ', dist)
    vmin = np.min(sig)
    vmax = np.max(sig)
    im = axs[0][n].matshow(data_covs[targ], vmin=vmin, vmax=vmax)
    axs[1][n].matshow(sig, vmin=vmin, vmax=vmax)
    axs[0][n].set_title('Node '+str(n))
    fig.colorbar(im, ax=[axs[0][n], axs[1][n]], shrink=0.75)


for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])
# plt.tight_layout()
# cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.75)
axs[0][0].set_ylabel('Emperical')
axs[1][0].set_ylabel('Model')


# plt.show()

## plotting

# plt.figure()
# Q = np.array(gq.Q_list)
# if np.min(Q) < 0:
#     Q -= np.min(Q)
# plt.plot(Q)

# plt.figure()
# plt.semilogy(Q)

plt.figure()
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.plasma(np.linspace(0,1,9)))
plt.plot(np.array(dists).T)
# plt.show()

plt.figure()
axs = plt.gca()
axs.scatter(data[:i+1+M+step,0], data[:i+1+M+step,1], color='m')
axs.plot(data[i+M+step-1:i+1+M+step,0], data[i+M+step-1:i+1+M+step,1], lw=2, color='b')
for n in np.arange(N):
    if n in gq.dead_nodes: # or (gq.n_obs[n] < 0.08):
# #         ## don't plot dead nodes
        axs.text(gq.mu[n,0]+0.01, gq.mu[n,1], s=str(n)+' (X)')
    else:
        el = np.linalg.inv(gq.L[n][:2,:2])
        sig = el.T @ el
        u,s,v = np.linalg.svd(sig)
        width, height = np.sqrt(s[0]*9), np.sqrt(s[1]*9) #*=4
        # if width>1e5 or height>1e5:
        #     pass
        # else:
        angle = atan2(v[0,1],v[0,0])*360 / (2*np.pi)
        # breakpoint()
        el = Ellipse((gq.mu[n,0],gq.mu[n,1]), width, height, angle, zorder=8)
        el.set_alpha(0.2)
        el.set_clip_box(axs.bbox)
        el.set_facecolor('r')  ##ed6713')
        axs.add_artist(el)
    
        axs.text(gq.mu[n,0]+0.01, gq.mu[n,1], s=str(n))
axs.scatter(gq.mu[:,0], gq.mu[:,1], c='k' , zorder=10)

u,s,v = np.linalg.svd(gq.obs.cov)
width, height = np.sqrt(s[0]*9), np.sqrt(s[1]*9) #*=4
# if width>1e5 or height>1e5:
#     pass
# else:
angle = atan2(v[0,1],v[0,0])*360 / (2*np.pi)
# breakpoint()
el = Ellipse((gq.obs.mean[0],gq.obs.mean[1]), width, height, angle, zorder=8)
el.set_alpha(0.1)
el.set_clip_box(axs.bbox)
el.set_facecolor('b')  ##ed6713')
axs.add_artist(el)
axs.scatter(gq.obs.mean[0], gq.obs.mean[1], c='k' , zorder=10)

# mu0 = np.array(gq.mu0_list)
# plt.plot(mu0[:,0], mu0[:,1], '--', color='b')
# plt.scatter(mu0[-1,0], mu0[-1,1], color='g')


plt.ylim([-50,50])
plt.xlim([-50,50])

plt.show()
# plt.figure()
# plt.plot(np.array(gq.pred))
# var_tmp = np.convolve(np.array(gq.pred), np.ones(500)/500, mode='valid')
# plt.plot(var_tmp, 'k')
# # for tt in gq.teleported_times:
# #     plt.axvline(x=tt, color='r', lw=1)

# plt.show()

print('----------------')

breakpoint()

