import time
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pylab as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D

from scipy.integrate import solve_ivp
from scipy.stats import multivariate_normal as mvn

from field.gqds_curr import GQDS
from datagen.diffeq import vanderpol, lorenz
from field.utils import center_mass
from datagen import plots

from math import atan2, floor


## parameters
## TODO: change def since not meshgrid-ing?

num_d = 22
d = 3
N = num_d**d
step = 8e-2
lam = 1e0 #-2
nu = 5 #1e-2
sigma_scale = 1e4 #1e4
mu_scale = 1
eps = 0

M = 10
T = 200    #500 + M
P = 0 

t_wait = 10 #M #100
B_thresh = -10 #-5.8
n_thresh = 5e-4

gq = GQDS(N, num_d, d, step=step, lam=lam, M=M, eps=eps, sigma_scale=sigma_scale, mu_scale=mu_scale, nu=nu, t_wait=t_wait, B_thresh=B_thresh, n_thresh=n_thresh)

## Get data from lorenz dataset
if T>1:
    # if d==2:
    #     x0, y0 = (0.1, 0.1)
    #     dt = 0.1
    #     sln = solve_ivp(vanderpol, (0, T), (x0, y0), args=(), dense_output=True, rtol=1e-6)
    #     t = np.linspace(0, dt*T, T)
    #     data = sln.sol(t).T * 10
    #     np.random.seed(42)
    #     data += np.random.randn(data.shape[0],data.shape[1])*0.05*np.std(data)

    # elif d==3:
    #     x0, y0, z0 = (0, 1, 1.05)
    #     dt = 0.05
    #     sln = solve_ivp(lorenz, (0, T), (x0, y0, z0), args=(), dense_output=True, rtol=1e-6)
    #     t = np.linspace(0, dt*T*2, T*2)
    #     # data = sln.sol(t).T[T:,:]
    #     data = sln.sol(t).T[100:,:]
    #     np.random.seed(42)
    #     data += np.random.randn(data.shape[0], data.shape[1])*0.05*np.std(data)

    s = np.load('data/lorenz_100trajectories_3dim_0to2000_noise0.2.npz')
    data = s['y'][0]    # or grab 2 trajectories as example
    # breakpoint()
    # data = np.vstack((s['y'][0], s['y'][1]))

    ## initialize things
    for i in np.arange(M):
        gq.observe(data[i])

gq.init_nodes()
print('Nodes initialized')

# Visualization
make_movie = True

if make_movie:
    ## Plotting during mesh refinement
    # fig, axs = plt.subplots(ncols=2, figsize=(6, 3), dpi=100)
    fig = plt.figure()
    axs = plt.gca(projection='3d')
    axs.view_init(40,23)
    # parameters for animation
    sweep_duration = 15
    hold_duration = 10
    total_duration = sweep_duration + hold_duration
    fps = 15

    # setup animation writer
    import matplotlib.animation
    writer_class = matplotlib.animation.writers['ffmpeg']
    writer = writer_class(fps=fps, bitrate=1000)
    writer.setup(fig, 'gqds_movie_3d.mp4')

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

# ## for plotting 3D ellipsoids
# var_v = np.linspace(0, np.pi, 10)
# var_u = np.linspace(0, 2 * np.pi, 10)
# x = np.outer(np.cos(var_u), np.sin(var_v))
# y = np.outer(np.sin(var_u), np.sin(var_v))
# z = np.outer(np.ones_like(var_u), np.cos(var_v))
# sphere = np.stack((x, y, z), 0).reshape(3,-1)


# Set of all spherical angles to draw our ellipsoid
n_points = 10
theta = np.linspace(0, 2*np.pi, n_points)
phi = np.linspace(0, np.pi, n_points)

# Get the xyz points for plotting
# Cartesian coordinates that correspond to the spherical angles:
X = np.outer(np.cos(theta), np.sin(phi))
Y = np.outer(np.sin(theta), np.sin(phi)).flatten()
Z = np.outer(np.ones_like(theta), np.cos(phi)).flatten()
old_shape = X.shape
X = X.flatten()

# es = np.random.normal(size=(200,3))
# es /= np.linalg.norm(es, axis=1)[...,None]

# x, y = np.meshgrid(es[:,0], es[:,1])

## run online
timer = time.time()
times = []
times_obs = []
for i in np.arange(-M, T - M):
    # print(i)
    t1 = time.time()
    gq.observe(data[i+M])
    times_obs.append(time.time()-t1)
    gq.em_step()    
    times.append(time.time()-t1)    

    if make_movie:
        axs.plot(data[:i+1+M,0], data[:i+1+M,1], data[:i+1+M,2], lw=2)
        if (i+M)>1900: #i < 200 or i > 300:
            # plots.plot3d_color(data[:i+1+M], t[:i+1+M], axs, alpha=1) #, cmap='PuBu')
            for n in np.arange(N):
                if n in gq.dead_nodes:
            # #         ## don't plot dead nodes
                    pass
                else:
                    el = np.linalg.inv(gq.L[n]).T
                    sig = el @ el.T
                    # Find and sort eigenvalues to correspond to the covariance matrix
                    eigvals, eigvecs = np.linalg.eigh(sig)
                    idx = np.sum(sig,axis=0).argsort()
                    eigvals_temp = eigvals[idx]
                    idx = eigvals_temp.argsort()
                    eigvals = eigvals[idx]
                    eigvecs = eigvecs[:,idx]

                    # Width, height and depth of ellipsoid
                    nstd = 2
                    rx, ry, rz = nstd * np.sqrt(eigvals)

                    # Rotate ellipsoid for off axis alignment
                    a,b,c = np.matmul(eigvecs, np.array([X*rx,Y*ry,Z*rz]))
                    a,b,c = a.reshape(old_shape), b.reshape(old_shape), c.reshape(old_shape)

                    # Add in offsets for the mean
                    a = a + gq.mu[n,0]
                    b = b + gq.mu[n,1]
                    c = c + gq.mu[n,2]
                    
                    axs.plot_surface(a, b, c, color='r', alpha=0.3)
            # breakpoint()

            #         ell = np.squeeze(el.dot(es[...,None])).T  * 2.25   # ?
            #         ell += gq.mu[n]
            #         axs.plot_trisurf(ell[:,0], ell[:,1], ell[:,2], color='r', alpha=0.3)
            #         ## 3D
            #         ell = (gq.fullSigma[n] @ sphere + gq.mu[n][...,None]).reshape(3,*x.shape)
            #         axs.plot_surface(*ell, color='r', alpha=0.3)
            #         # print('------------------- ', i+M, np.max(gq.fullSigma[n]))
            #         if np.max(np.abs(gq.fullSigma[n])) > 6:
            #             print('Node ', n, ' has large sigma at time ', i+M)
            #             print('Any dead nodes available? ', gq.dead_nodes)
                    ## 2D
                    # u,s,v = np.linalg.svd(gq.fullSigma[n])
                    # width, height = s[0]*2.25, s[1]*2.25 #*=4
                    # if width>1e5 or height>1e5:
                    #     pass
                    # else:
                    #     angle = atan2(v[0,1],v[0,0])*360 / (2*np.pi)
                    #     # breakpoint()
                    #     el = Ellipse((gq.mu[n,0],gq.mu[n,1]), width, height, angle, zorder=8)
                    #     el.set_alpha(0.2)
                    #     el.set_clip_box(axs.bbox)
                    #     el.set_facecolor('r')  ##ed6713')
                    #     axs.add_artist(el)
                
                    # axs.text(gq.mu[n,0]+0.1, gq.mu[n,1], gq.mu[n,2], s=str(n))

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

        axs.view_init(40,23)

        # plt.xlim([0,1600])
        # plt.ylim([-25,15])
        # axs.set_zlim([-1550,0])

        mask = np.ones(gq.mu.shape[0], dtype=np.bool)
        if gq.dead_nodes:
            mask[np.array(gq.dead_nodes)] = False
        # mask[gq.n_obs<1e-8] = False
        # breakpoint()
        axs.scatter(gq.mu[mask,0], gq.mu[mask,1], gq.mu[mask,2], c='k' , zorder=10)

        # axs.set_xticks([-15,15])
        # axs.set_yticks([-15,20])
        # axs.set_zticks([0,25,50])
        axs.set_xticks([])
        axs.set_yticks([])
        axs.set_zticks([])
        
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

        if i+M in [M-1, floor(T/4)-1, floor(T/2)-1, T-1]:
            figS = plt.gcf()
            figS.savefig('lorenz_'+str(i+1)+'.svg', bbox_inches='tight')
        
        # axs[0].cla()
        # axs[1].cla()
        axs.cla()


    # print(i)
    if (i+M) % 50 == 0  and i>0:
        print(i+M, 'frames processed. Time elapsed ', time.time()-timer)

print('Done fitting all data online')
print('Average cycle time: ', np.mean(np.array(times)[20:]))
print('Average observation time: ', np.mean(np.array(times_obs)[20:]))
print('Average prediction time: ', np.mean(np.array(gq.time_pred)[20:]))


if make_movie:
    writer.finish()

## plotting

# import networkx as nx
# G = nx.Graph(gq.neighbors)
# nx.draw(G, gq.mu)
# plt.show()

plt.figure()
Q = np.array(gq.Q_list)
if np.min(Q) < 0:
    Q -= np.min(Q)
plt.plot(Q)

plt.figure()
plt.semilogy(Q)


plt.figure()
plt.plot(np.array(gq.pred))
for tt in gq.teleported_times:
    plt.axvline(x=tt, color='r', lw=1)

plt.show()

print('----------------')

breakpoint()