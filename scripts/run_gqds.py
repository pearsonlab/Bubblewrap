import time
import numpy as np
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
N = 25
d = 2
step = 5e-1
lam = 2
nu = 2
rep = 1
sigma_scale = 9e3 #9e3 #1.5e4 #9e3

T = 400
dt = 0.1
M = 30

gq = GQDS(N, d, step=step, lam=lam, M=M, sigma_scale=sigma_scale, nu=nu)

## Generate data from 2d vdp oscillator
x0, y0 = (0.1, 0.1)
dim = 2
sln = solve_ivp(vanderpol, (0, T), (x0, y0), args=(), dense_output=True, rtol=1e-6)
t = np.linspace(0, dt*T, T)
data = sln.sol(t).T * 10

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
    sweep_duration = 20
    hold_duration = 10
    total_duration = sweep_duration + hold_duration
    fps = 20

    # setup animation writer
    import matplotlib.animation
    writer_class = matplotlib.animation.writers['ffmpeg']
    writer = writer_class(fps=fps, bitrate=2000)
    writer.setup(fig, 'gqds_movie.mp4')

size = 10

com = center_mass(gq.mu)

max_x = np.max(np.abs(gq.mu_orig[:,0].flatten())) * 1.1
max_y = np.max(np.abs(gq.mu_orig[:,1].flatten())) * 1.1

x = np.linspace(-max_x, max_x, size)
y = np.linspace(-max_y, max_y, size)
x, y = np.meshgrid(x, y)
pos = np.dstack((x, y))


## run online
timer = time.time()
for i in np.arange(0, T - M):
    gq.observe(data[i+M])
    for _ in np.arange(rep):
        gq.em_step()
    
        if make_movie:
            for n in np.arange(N):
                u,s,v = np.linalg.svd(gq.fullSigma[n])
                width, height = s[0], s[1]
                angle = atan2(v[0,1],v[0,0])*360 / (2*np.pi)
                # breakpoint()
                el = Ellipse((gq.mu[n,0],gq.mu[n,1]), width, height, angle)
                el.set_alpha(0.1)
                el.set_clip_box(axs.bbox)
                el.set_facecolor('r')
                axs.add_artist(el)

            # plots.plot_color(data[:i+M+1, 0], data[:i+M+1, 1], t[:i+M+1], axs[0])
            # axs[0].scatter(gq.mu_orig[:,0], gq.mu_orig[:,1])
            plots.plot_color(data[:i+M+1, 0], data[:i+M+1, 1], t[:i+M+1], axs)
            axs.scatter(gq.mu[:,0], gq.mu[:,1], c='k')

            # for n in np.arange(N):
            #     rv = mvn(gq.mu[n], gq.fullSigma[n])
            #     z = rv.pdf(pos)
            #     m = np.amax(z) *0.33
            #     axs.contour(x,y,z, levels=[m], colors='k')

            # yl = axs[1].get_ylim()
            # xl = axs[1].get_xlim()
            # axs[0].set_ylim(yl)
            # axs[0].set_xlim(xl)


            plt.draw()
            writer.grab_frame()
            
            # axs[0].cla()
            # axs[1].cla()
            axs.cla()


    # print(i)
    if i % 500 == 0:
        print(i, 'frames processed. Time elapsed ', time.time()-timer)

print('Done fitting all data online')

if make_movie:
    writer.finish()

## plotting

cmaps = ['Blues', 'Reds']

print('----------------')
print(np.sum(gq.sigma-gq.sigma_orig))
print(np.sum(gq.mu-gq.mu_orig))
# print(np.sum(gq.A-gq.A_orig))
# breakpoint()
fig, axs = plt.subplots(ncols=2, figsize=(8, 4), dpi=200)
color=plt.cm.rainbow(np.linspace(0,1,gq.N))
for n in np.arange(N):

    rv = mvn(gq.mu_orig[n], gq.fullSigma_orig[n])
    z = rv.pdf(pos)
    # z /= np.sum(z)
    m = np.amax(z) *0.2
    # breakpoint()
    # z[z<1e-2] = 0
    axs[0].contour(x,y,z, levels=[m], colors='k') #color[n])

    rv = mvn(gq.mu[n], gq.fullSigma[n])
    z = rv.pdf(pos)
    # z /= np.sum(z)
    m = np.amax(z) *0.2
    # breakpoint()
    # z[z<1e-2] = 0
    axs[1].contour(x,y,z, levels=[m], colors='k')

    # m = np.amax(z) / 2
    # step = m/2
    # levels = np.arange(0.0, m, step) + step
    # plt.contourf(x, y, z, levels, cmap=cmaps[1], extend='max', alpha=0.4)

    # z = (1/(2*np.pi*gq.sigma_orig[i]**2) * np.exp(-((x-gq.mu_orig[i][0])**2/(2*gq.sigma_orig[i]**2)
    #  + (y-gq.mu_orig[i][1])**2/(2*gq.sigma_orig[i]**2))))
    # m = np.amax(z) / 2
    # step = m/2
    # levels = np.arange(0.0, m, step) + step
    # plt.contourf(x, y, z, levels, cmap=cmaps[1], extend='max', alpha=0.2)

    # z = (1/(2*np.pi*gq.sigma[i]**2) * np.exp(-((x-gq.mu[i][0])**2/(2*gq.sigma[i]**2)
    #  + (y-gq.mu[i][1])**2/(2*gq.sigma[i]**2))))
    # m = np.amax(z) / 2
    # step = m/2
    # levels = np.arange(0.0, m, step) + step
    # plt.contourf(x, y, z, levels, cmap=cmaps[0], extend='max', alpha=0.2)

# axs = plt.gca()
axs[0].scatter(gq.mu_orig[:,0], gq.mu_orig[:,1], c='blue')
axs[1].scatter(gq.mu[:,0], gq.mu[:,1], c='orange')
plots.plot_color(data[:, 0], data[:, 1], t, axs[0])
plots.plot_color(data[:, 0], data[:, 1], t, axs[1])

# plt.colorbar()
plt.show()

breakpoint()


fig, axs = plt.subplots(ncols=2, figsize=(6, 3), dpi=100)
# axs = plt.gca()
plots.plot_color(data[:M, 0], data[:M, 1], t[:M], axs[0])
plots.plot_color(data[:, 0], data[:, 1], t, axs[1])
axs[0].scatter(gq.mu_orig[:,0], gq.mu_orig[:,1])
axs[1].scatter(gq.mu[:,0], gq.mu[:,1], c='orange')

yl = axs[1].get_ylim()
xl = axs[1].get_xlim()
axs[0].set_ylim(yl)
axs[0].set_xlim(xl)

# plt.figure()
# plt.plot(np.array(gq.Q_list))

plt.show()

breakpoint()