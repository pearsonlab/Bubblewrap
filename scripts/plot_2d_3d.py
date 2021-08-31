import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pylab as plt
from matplotlib.patches import Ellipse
from math import atan2, floor


fig = plt.figure(figsize=plt.figaspect(0.5))
axs = fig.add_subplot(1, 2, 1)

### 2D vdp oscillator
s = np.load('vdp_1trajectories_2dim_500to20500_noise0.05.npz')
data = s['y'][0]

A = np.load('vdp_2d_A.npy')
mu = np.load('vdp_2d_mu.npy')
L = np.load('vdp_2d_L.npy')
n_obs = np.load('vdp_2d_n_obs.npy')

pred = np.load('vdp_2d_pred.npy')
entropy = np.load('vdp_2d_entropy.npy')

axs.plot(data[:,0], data[:,1], color='gray', alpha=0.8)
for n in np.arange(A.shape[0]):
    if n_obs[n] > 0.2:
        el = np.linalg.inv(L[n])
        sig = el.T @ el
        u,s,v = np.linalg.svd(sig)
        width, height = np.sqrt(s[0])*3, np.sqrt(s[1])*3
        angle = atan2(v[0,1],v[0,0])*360 / (2*np.pi)
        el = Ellipse((mu[n,0], mu[n,1]), width, height, angle, zorder=8)
        el.set_alpha(0.4)
        el.set_clip_box(axs.bbox)
        el.set_facecolor('#ed6713')
        axs.add_artist(el)

mask = np.ones(mu.shape[0], dtype=bool)
mask[n_obs<1] = False
axs.scatter(mu[mask,0], mu[mask,1], c='k' , zorder=10)

axs.set_xticks([-2, -1, 0, 1, 2])
axs.set_yticks([-6, -3, 0, 3, 6])

in1, in2 = -0.15, 1
axs.text(in1, in2, s='a', transform=axs.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')


### 3D lorenz
axs = fig.add_subplot(1, 2, 2, projection='3d')

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


s = np.load('lorenz_1trajectories_3dim_500to20500_noise0.05.npz')
data = s['y'][0]

A = np.load('lorenz_3d_A.npy')
mu = np.load('lorenz_3d_mu.npy')
L = np.load('lorenz_3d_L.npy')
n_obs = np.load('lorenz_3d_n_obs.npy')

pred = np.load('lorenz_3d_pred.npy')
entropy = np.load('lorenz_3d_entropy.npy')

axs.plot(data[:,0], data[:,1], data[:,2], color='gray', alpha=0.8)
for n in np.arange(A.shape[0]):
    if n_obs[n] > 1e-4:
        el = np.linalg.inv(L[n]).T
        sig = el @ el.T
        # Find and sort eigenvalues to correspond to the covariance matrix
        eigvals, eigvecs = np.linalg.eigh(sig)
        idx = np.sum(sig,axis=0).argsort()
        eigvals_temp = eigvals[idx]
        idx = eigvals_temp.argsort()
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:,idx]

        # Width, height and depth of ellipsoid
        nstd = 3
        rx, ry, rz = nstd * np.sqrt(eigvals)

        # Rotate ellipsoid for off axis alignment
        a,b,c = np.matmul(eigvecs, np.array([X*rx,Y*ry,Z*rz]))
        a,b,c = a.reshape(old_shape), b.reshape(old_shape), c.reshape(old_shape)

        # Add in offsets for the mean
        a = a + mu[n,0]
        b = b + mu[n,1]
        c = c + mu[n,2]
        
        axs.plot_surface(a, b, c, color='#ff4400', alpha=0.6)

axs.view_init(40,23)

mask = np.ones(mu.shape[0], dtype=bool)
mask[n_obs<1e-4] = False
axs.scatter(mu[mask,0], mu[mask,1], mu[mask,2], c='k' , zorder=10)

axs.set_xticks([200, 600, 1000, 1400])
axs.set_yticks([-20, -10, 0, 10])
axs.set_zticks([-1400, -1000, -600, -200])
in1, in2 = 0, 1
axs.text(in1, in2, 100, s='b', transform=axs.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')


# plt.tight_layout()
# fig.savefig('Fig2_b.png', bbox_inches='tight')

plt.show()