#%%
import time
from math import atan2
import numpy as np
from numpy.lib.ufunclike import _fix_and_maybe_deprecate_out_named_y
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import multivariate_normal as mvn

import matplotlib
import matplotlib.pylab as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse

from field.gqds import GQDS, get_fullSigma
from datagen.diffeq import vanderpol
from field.utils import center_mass
from datagen.plots import draw_bubbles, draw_graph, plot_scatter_connected
from datagen import plots

import jPCA
from jPCA.util import load_churchland_data, plot_projections

from proSVD import proSVD

import networkx as nx

#%% load jPCA data
# data = projected
path = "/hdd/pgupta/"
dataloc = path + 'jpca_full.npy'
dataloc_jpcaed = path + 'jpca_reduced.npy'
data = np.load(dataloc)
data_jpcaed = np.load(dataloc_jpcaed)
# trials, timesteps, n = data.shape
# (trials, window, components) = data_jpcaed.shape

timesteps = 21
# data = data[:, :timesteps, :]
# data = data.reshape((trials*timesteps, n))
# data_jpcaed = data_jpcaed.reshape((trials*window, components))

# repeats = 2
# total = timesteps * 108 #int(data.shape[0] / 2)
# data = data[:total, :]
data = data_jpcaed[:, :].T
# data = np.fliplr(data, axis=0)
# # repeating data
# for i in range(repeats):
    # data = np.append(data, data, axis=0)
#%%
def plot_jPCA_trials(t, timesteps, data, ax=None, alpha=.1, color='k'):
    full_trials = np.floor( t / timesteps).astype('int')

    ind = 0
    # colors = cm.rainbow(np.linspace(0,1,full_trials))
    for j in range(full_trials):
        ax.plot(data[ind:ind+timesteps, 0], data[ind:ind+timesteps, 1], 
                alpha=alpha, color=color, zorder=10) #colors[j])
        # ax.scatter(data[ind, 0], data[ind, 1], alpha=0.1)
        ind += timesteps

def ewma(data, com):
    return np.array(pd.DataFrame(data=dict(data=data)).ewm(com).mean()['data'])

def get_fullSigma(L):
    inv = np.linalg.inv(L)
    return inv.T @ inv


#%% params
make_movie = False

# general params
dt = 0.1
M = timesteps*4 # 4 trials
T = data.shape[0] - M # iters
t = np.linspace(0, dt*T, T)
k = 6

# ssSVD params
ssSVD = False
trueSVD = False
if ssSVD:
    l = 1 # should be able to handle this
    l1 = k
    decay = 1
    pro = proSVD(k=k, decay_alpha=decay, 
                 trueSVD=trueSVD, history=0)

# bubblewrap params
d = k
num_d = 3
N = 100
lam = 1e-3
nu = 1e-5

step = 9e-2
# sigma_scale = 1e3
B_thresh = -10
n_thresh = 5e-4
eps = 0

P = 0 
t_wait = 1 # breadcrumbing
gq = GQDS(N, num_d, d, step=step, lam=lam, eps=eps, M=M,
          nu=nu, t_wait=t_wait, n_thresh=n_thresh, B_thresh=B_thresh)

##%% init and run
if ssSVD:
    # init with just regular svd
    pro.initialize(data[:max(k, M), :].T)
    data_red = data[:max(k, M)] @ pro.Q
    for i in np.arange(M):
        gq.observe(data_red[i])
else:
    for i in np.arange(M):
        gq.observe(data[i])
        

gq.init_nodes()
print('Nodes initialized')
gq.printing = False
# breakpoint()

# visualize
if make_movie:
    ## Plotting during mesh refinement
    # fig, axs = plt.subplots(ncols=2, figsize=(6, 3), dpi=100)
    fig = plt.figure()
    axs = plt.gca()
    # parameters for animation
    sweep_duration = 15
    hold_duration = 10
    total_duration = sweep_duration + hold_duration
    fps = 30

    # setup animation writer
    import matplotlib.animation
    writer_class = matplotlib.animation.writers['ffmpeg']
    writer = writer_class(fps=fps, bitrate=1000)
    writer.setup(fig, 'gqds_movie.mp4')
else:
    fig = plt.figure(figsize=(14,8))
    trials = [5, 10, 25, 50, 100]
    ts_to_plot = timesteps * np.array(trials)
    ax_ind = 1

size = 10

com = center_mass(gq.mu)

max_x = np.max(np.abs(gq.mu[:,0].flatten())) * 1.1
max_y = np.max(np.abs(gq.mu[:,1].flatten())) * 1.1

x = np.linspace(-max_x, max_x, size)
y = np.linspace(-max_y, max_y, size)
x, y = np.meshgrid(x, y)
pos = np.dstack((x, y))

## %% run online
if ssSVD:
    data_red_all = np.zeros((data.shape[0], k))
    data_red_all[:max(M, k), :] = data_red
else:
    data_red_all = data


for i in np.arange(-M, T - M):
    t1 = time.time()
    if ssSVD:
        pro.updateSVD(data[i+M, np.newaxis].T)
        data_red = pro.Q.T @ data[i+M].T
        if data_red.sum() == 0:
            breakpoint()
        data_red_all[i+(2*M), :] = data_red
        gq.observe(data_red)
        gq.em_step()
    else:
        gq.observe(data[i+M])
        gq.em_step()
        for j in range(5):
            gq.grad_Q()

    if i % 75 == 0:
        print(gq.pred[i+M])    

    if make_movie:
        if True: #i < 200 or i > 300:
            plot_jPCA_trials(data, ax=axs)
            for n in np.arange(N):
                if n in gq.dead_nodes:
                    ## don't plot dead nodes
                    pass
                else:
                    # sig = gq.L[n] @ gq.L[n].T
                    u,s,v = np.linalg.svd(gq.L[n])
                    width, height = s[0]*2.25, s[1]*2.25 #*=4
                    if width>1e5 or height>1e5:
                        pass
                    else:
                        angle = atan2(v[0,1],v[0,0])*360 / (2*np.pi)
                        # breakpoint()
                        el = Ellipse((gq.mu[n,0],gq.mu[n,1]), width, height, angle, zorder=8)
                        el.set_alpha(0.2)
                        el.set_clip_box(axs.bbox)
                        el.set_facecolor('r')  ##ed6713')
                        axs.add_artist(el)
                
                    # plt.text(gq.mu[n,0]+1, gq.mu[n,1], str(n))

        else: #i between 200 and 300
            # find node closest to data point
            node = np.argmax(gq.alpha)
            A = gq.A.copy()
            A[A<=(1/N)] = 0
            plots.plot_color(data[:i+M+1, 0], data[:i+M+1, 1], t[:i+M+1], axs, alpha=1) #, cmap='PuBu')
            for j in np.arange(N):
                if A[node,j] > 0 and not node==j:
                    print('Arrow from ', str(node), ' to ', str(j))
                    axs.arrow(gq.mu[node,0], gq.mu[node,1], gq.mu[j,0]-gq.mu[node,0], gq.mu[j,1]-gq.mu[node,1], length_includes_head=True, width=A[node,j], head_width=0.8, color='k', zorder=9)

        axs.scatter(gq.mu[:,0], gq.mu[:,1], c='k' , zorder=10)
        plt.draw()
        writer.grab_frame()

        if i >= 200 and i <= 300:
            writer.grab_frame()
        
        # axs[0].cla()
        # axs[1].cla()
        axs.cla()
    else:
        if i in ts_to_plot:
            ax = fig.add_subplot(2, 3, ax_ind)
            plot_jPCA_trials(i+M, timesteps, data_red_all, ax=ax)
            draw_bubbles(gq, sig_ell=2, ax=ax)
            # draw_graph(ax, gq, thresh=.0103)
            # plot_jPCA_trials(data, ax=ax)
            ax.set(title='trial {} ({:.2f} seconds)'.format(trials[ax_ind-1], i*.01))
            ax_ind += 1

xmin, ymin = data_red_all.min(axis=0)[:2]
xmax, ymax = data_red_all.max(axis=0)[:2]

for currax in fig.axes:
    currax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
if make_movie:
    writer.finish()

ax = fig.add_subplot(2, 3, ax_ind)
ax.plot(gq.pred, color='grey', alpha=0.2)
ax.plot(ewma(gq.pred, 100), color='black')
ax.axvline(timesteps*100)
ax.set(xlabel='timesteps', ylabel='log probability')

fig.suptitle('Bubblewrap approximates jPCA trajectories within ~10 trials', y=.95)
fig.subplots_adjust(wspace=0.3)
# plt.savefig('jpca_fig.pdf', bbox_inches='tight')

#%% feed gq "new" trajectories - ORDER MATTERS!
trials = 45, 27

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9,4))
for trial in trials:
    start, end = int(timesteps*trial), int(timesteps*(trial+1))
    newtrial = data[start:end, :]
    closest_nodes = []
    for i in range(newtrial.shape[0]):
        gq.observe(newtrial[i])
        gq.em_step()
        closest_nodes.append(gq.current_node)
    closest_nodes = np.array(closest_nodes, dtype='int')
    nodes_to_bubble = np.unique(closest_nodes)

    # plot 1 - all trials, all bubbles
    draw_bubbles(gq, sig_ell=2, ax=ax[0])
    plot_jPCA_trials(data.shape[0], timesteps, data, ax=ax[0], alpha=.03)
    ax[0].plot(newtrial[:, 0], newtrial[:, 1], lw=2, color='blue')
    # draw_graph(gq, ax=ax[0], thresh=1/N)

    # plot 2 - one trial, closest bubbles, graph
    A = np.copy(np.array(gq.A))
    A_mask = np.ones(A.shape) # all true (to discard)
    A_mask[nodes_to_bubble, :] = np.zeros(A_mask.shape[1]) # keep closest nodes
    A_mask = A_mask.astype('bool')
    A[A_mask] = 0

    draw_graph(gq, ax[1], A=A, thresh=1/N, node_color='orange')
    draw_bubbles(gq, sig_ell=2, ax=ax[1], 
                node_list=nodes_to_bubble)
    plot_jPCA_trials(data.shape[0], timesteps, data, ax=ax[1], alpha=.03)
    ax[1].plot(newtrial[:, 0], newtrial[:, 1], lw=2, color='blue')

# %%

# %%
# datas, times = load_churchland_data(path)
# num_jpcs = 2

# jpca = jPCA.JPCA(num_jpcs=num_jpcs)
# # Fit the jPCA object to data
# (projected, 
#  full_data_var,
#  pca_var_capt,
#  jpca_var_capt) = jpca.fit(datas, times=times, tstart=-50, tend=150)

# data = np.array(projected).reshape((len(projected)*projected[0].shape[0], num_jpcs))
# data = data[:T, :]
# # data += np.random.randn(data.shape[0],data.shape[1])*0.05*np.std(data)



# %%
