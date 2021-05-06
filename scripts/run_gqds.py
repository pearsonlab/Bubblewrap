import time
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pylab as plt
from matplotlib.patches import Ellipse

from scipy.integrate import solve_ivp
from scipy.stats import multivariate_normal as mvn

from field.gqds import GQDS
from datagen.diffeq import vanderpol
from field.utils import center_mass
from datagen import plots

from math import atan2


## parameters
N = 64
d = 2
step = 1e-1
lam = 2
nu = 2
sigma_scale = 1e4 #9e3 #1.5e4 #9e3

dt = 0.1
M = 30
T = 300 + M
P = 0 #20 #200

gq = GQDS(N, d, step=step, lam=lam, M=M, sigma_scale=sigma_scale, nu=nu) #, beta=beta)

## Generate data from 2d vdp oscillator
if T>1:
    x0, y0 = (0.1, 0.1)
    dim = 2
    sln = solve_ivp(vanderpol, (0, T), (x0, y0), args=(), dense_output=True, rtol=1e-6)
    t = np.linspace(0, dt*T, T)
    data = sln.sol(t).T * 10
    np.random.seed(42)
    data += np.random.randn(data.shape[0],data.shape[1])*0.05*np.std(data)

    ## initialize things
    for i in np.arange(M):
        gq.observe(data[i])

gq.init_nodes()
print('Nodes initialized')
# breakpoint()

# Visualization
make_movie = True

if make_movie:
    ## Plotting during mesh refinement
    # fig, axs = plt.subplots(ncols=2, figsize=(6, 3), dpi=100)
    fig = plt.figure()
    axs = plt.gca()
    # parameters for animation
    sweep_duration = 15
    hold_duration = 10
    total_duration = sweep_duration + hold_duration
    fps = 15

    # setup animation writer
    import matplotlib.animation
    writer_class = matplotlib.animation.writers['ffmpeg']
    writer = writer_class(fps=fps, bitrate=1000)
    writer.setup(fig, 'gqds_movie.mp4')

size = 10

com = center_mass(gq.mu)

max_x = np.max(np.abs(gq.mu_orig[:,0].flatten())) * 1.1
max_y = np.max(np.abs(gq.mu_orig[:,1].flatten())) * 1.1

x = np.linspace(-max_x, max_x, size)
y = np.linspace(-max_y, max_y, size)
x, y = np.meshgrid(x, y)
pos = np.dstack((x, y))


## settle mesh
# for i in np.arange(P):
#     print(i)
#     gq.grad_Q()  
#     if make_movie: #i < 200 or i > 300:
#         # plots.plot_color(data[:i+M+1, 0], data[:i+M+1, 1], t[:i+M+1], axs, alpha=1) #, cmap='PuBu')
#         for n in np.arange(N):
#             inv = np.linalg.inv(gq.L[n])
#             sig = inv.T @ inv
#             u,s,v = np.linalg.svd(sig)
#             width, height = s[0]*2.25, s[1]*2.25 #*=4
#             if width>1e5 or height>1e5:
#                 pass
#             else:
#                 angle = atan2(v[0,1],v[0,0])*360 / (2*np.pi)
#                 # breakpoint()
#                 el = Ellipse((gq.mu[n,0],gq.mu[n,1]), width, height, angle, zorder=8)
#                 el.set_alpha(0.2)
#                 el.set_clip_box(axs.bbox)
#                 el.set_facecolor('r')  ##ed6713')
#                 axs.add_artist(el)
#         axs.scatter(gq.mu[:,0], gq.mu[:,1], c='k' , zorder=10)

#         # plt.gca().set_xticks([],[])
#         # plt.gca().set_yticks([],[])

#         plt.xlim([-30,30])
#         plt.ylim([-30,30])

#         plt.draw()
#         writer.grab_frame()
#         axs.cla()

## run online
timer = time.time()
times = []
for i in np.arange(0, T - M):
    # print(i)
    t1 = time.time()
    gq.observe(data[i+M])
    gq.em_step()    
    times.append(time.time()-t1)    

    if make_movie:
        if True: #i < 200 or i > 300:
            plots.plot_color(data[:i+1+M, 0], data[:i+1+M, 1], t[:i+1+M], axs, alpha=1) #, cmap='PuBu')
            for n in np.arange(N):
                # sig = gq.L[n] @ gq.L[n].T
                u,s,v = np.linalg.svd(gq.fullSigma[n])
                width, height = s[0]*4, s[1]*4 #*=4
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
            # plots.plot_color(data[:i+M+1, 0], data[:i+M+1, 1], t[:i+M+1], axs, alpha=1) #, cmap='PuBu')
            for j in np.arange(N):
                if A[node,j] > 0 and not node==j:
                    print('Arrow from ', str(node), ' to ', str(j))
                    axs.arrow(gq.mu[node,0], gq.mu[node,1], gq.mu[j,0]-gq.mu[node,0], gq.mu[j,1]-gq.mu[node,1], length_includes_head=True, width=A[node,j], head_width=0.8, color='k', zorder=9)

        plt.xlim([-30,30])
        plt.ylim([-30,30])

        axs.scatter(gq.mu[:,0], gq.mu[:,1], c='k' , zorder=10)

        # plt.gca().set_xticks([],[])
        # plt.gca().set_yticks([],[])
        
        # for n in np.arange(N):
        #     rv = mvn(gq.mu[n], gq.fullSigma[n])
        #     z = rv.pdf(pos)
        #     m = np.amax(z) *0.33
        #     axs.contour(x,y,z, levels=[m], colors='k')

        # yl = axs[1].get_ylim()
        # xl = axs[1].get_xlim()
        # axs[0].set_ylim(yl)
        # axs[0].set_xlim(xl)

        ## networkxx nx
        # G = nx.Graph(gq.neighbors)
        # nx.draw(G, gq.mu)
        # plt.show()


        plt.draw()
        writer.grab_frame()

        # if i >= 200 and i <= 300:
        #     writer.grab_frame()
        
        # axs[0].cla()
        # axs[1].cla()
        axs.cla()


    # print(i)
    if i % 100 == 0:
        print(i, 'frames processed. Time elapsed ', time.time()-timer)

print('Done fitting all data online')
print('Average cycle time: ', np.mean(np.array(times)[1:]))

if make_movie:
    writer.finish()

## plotting

# import networkx as nx
# G = nx.Graph(gq.neighbors)
# nx.draw(G, gq.mu)
# plt.show()

print('----------------')

breakpoint()