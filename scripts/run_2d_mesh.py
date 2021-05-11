import time
import numpy as np
import networkx as nx
from field.utils import dumb_bounding

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

import matplotlib.pylab as plt
from datagen import plots
from datagen.models import lorenz, vanderpol
from field.mesh import Mesh
from scipy.integrate import solve_ivp

## Define parameters

# Datagen
T = 1000
dt = 0.1
M = 50

# Mesh
num = 10
spr = 1
max_neighb = 8

# Optimization
step = 1e-4
seed = 42
internal_reps = 20

# Visualization
make_movie = True

## Generate data from 2d vdp oscillator
x0, y0 = (0.1, 0.1)
dim = 2
sln = solve_ivp(vanderpol, (0, T), (x0, y0), args=(), dense_output=True, rtol=1e-6)
t = np.linspace(0, dt*T, T)
data = sln.sol(t).T * 100 #scale makes for easier human readability imo

if make_movie:
    ## Plotting during mesh refinement
    fig, axs = plt.subplots(ncols=3, figsize=(12, 4), dpi=200)
    # parameters for animation
    sweep_duration = 20
    hold_duration = 10
    total_duration = sweep_duration + hold_duration
    fps = 20

    # setup animation writer
    import matplotlib.animation
    writer_class = matplotlib.animation.writers['ffmpeg']
    writer = writer_class(fps=fps, bitrate=2000)
    writer.setup(fig, 'mesh_movie.mp4')

## Initialize mesh [around data]
mesh = Mesh(num, dim=dim, M=M, mesh_type='triangle')

## Give first few obs without doing anything
mesh.quiet(data[:M, :])
mesh.initialize_mesh()
# mesh.jax_relax()

timers = []
init_time = time.time()

for i in np.arange(0, T - M):
    # get new observation
    mesh.observe(data[i+M])
    timer = time.time()
    for j in np.arange(0,internal_reps):
        
        
        # spatial/vector gradient updates
        # mesh.grad_pred()
        # mesh.jax_grad()

        mesh.full_grad()

        # adjust vectors of bounding points
        # mesh.grad_vec()
        # spring relaxation gradient update
        # mesh.relax_network()
        # mesh.jax_relax()
        # m = np.sum(mesh.coords0 - mesh.coords, axis=1) != 0
        # v = np.sum(mesh.vectors0 - mesh.vectors, axis=1) != 0

    if make_movie:
        plots.plot_color(data[:i+M+1, 0], data[:i+M+1, 1], t[:i+M+1], axs[0])
        axs[0].quiver(mesh.coords0[:, 0], mesh.coords0[:, 1], mesh.vectors0[:, 0], mesh.vectors0[:, 1])
        plots.plot_color(data[:i+M+1, 0], data[:i+M+1, 1], t[:i+M+1], axs[1])
        axs[1].quiver(mesh.coords[:, 0], mesh.coords[:, 1], mesh.vectors[:, 0], mesh.vectors[:, 1], color='k')
        plots.plot_color(data[:i+M+1, 0], data[:i+M+1, 1], t[:i+M+1], axs[2])
        curr_pos = dict((i,c.tolist()) for i,c in enumerate(mesh.coords))
        nx.draw(mesh.G, curr_pos, node_size=1)
        plt.draw()
        writer.grab_frame()
        
        axs[0].cla()
        axs[1].cla()
        axs[2].cla()
    
    timers.append(time.time()-timer)

    if i% 100 == 0:
        print(i, ' data points processed; Time elapsed: ', time.time()-init_time)

    # plt.show()

_, mesh.bounding = dumb_bounding(mesh.coords, mesh.obs.mid, num=2**mesh.d)
print('a mean ', np.mean(mesh.a))

print('Average cycle time ', np.mean(timers))

if make_movie:
    writer.finish()

###### 2d plots
if not make_movie:
    fig, axs = plt.subplots(ncols=3, figsize=(15, 5), dpi=300)

bp = mesh.bounding

plots.plot_color(data[:M, 0], data[:M, 1], t, axs[0])
axs[0].quiver(mesh.coords0[:, 0], mesh.coords0[:, 1], mesh.vectors0[:, 0], mesh.vectors0[:, 1])
# plots.plot_color(data[:, 0], data[:, 1], t, axs[1])
axs[1].quiver(mesh.coords[:, 0], mesh.coords[:, 1], mesh.vectors[:, 0], mesh.vectors[:, 1], color='k')
plots.plot_color(data[:, 0], data[:, 1], t, axs[1])
# mid = np.asarray(mesh.obs.mid_list)
# vect = np.asarray(mesh.obs.vect_list)
# axs[1].quiver(mid[:, 0], mid[:, 1], vect[:, 0], vect[:, 1])

axs[0].title.set_text('Initial grid (random)')
axs[1].title.set_text('Final grid, 1 step/new observation')

plots.plot_color(data[:, 0], data[:, 1], t, axs[2])
curr_pos = dict((i,c.tolist()) for i,c in enumerate(mesh.coords))
nx.draw(mesh.G, curr_pos, node_size=1)

plt.autoscale()
plt.show()