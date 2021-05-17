import numpy as np

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors # for colorbar scale

#%% load distortion, timing data for RP -> ssSVD 
npz = np.load('neurips/rp_dist_timing.npz')
rp_range = npz['rp_range']
accs = npz['accs']
times = npz['times']
baseline_svd = npz['baseline_svd']
baseline_rp = npz['baseline_rp']
baseline_svd_time = npz['baseline_svd_time']
baseline_rp_time = npz['baseline_rp_time']
N = 10000  # original dim

#%% load ssSVD on monkey reach data
npz = np.load('neurips/ssSVD_results.npz')
l1 = npz['l1']
k = npz['k']
spikes = npz['spikes']
bin_size = npz['bin_size']
Qs = npz['Qs']
Us = npz['Us']

# %% total plot
# setting up gridspec - top row and two bottom rows
fig = plt.figure(figsize=(12, 9))
plt.subplots_adjust(hspace=0.38, wspace=0.30)

outer = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
gs1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0])
gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs1[0])
gs_bottom = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec = outer[1],
                                             hspace=0.15)  # less space betwen bottom 2 rows

# ---------------- TOP ROW ------------------ #
cmap = cm.get_cmap('coolwarm') # change this? 
c0 = cmap(.95) # for svd baseline
c1 = cmap(0) # for rp baseline

labels = ['RP $N$ to $n$, \n SVD $n$ to $k$', 
          'SVD $N$ to $k$', 
          'RP $N$ to $k$']

panel = 97 # ascii for a
in1, in2 = -0.06, 1.15 # panel labels position

# distortion
ax = plt.subplot(gs_top[0])
ax.text(in1, in2, chr(panel), transform=ax.transAxes, 
        fontsize=16, fontweight='bold', va='top', ha='right')
g = ax.errorbar(rp_range[:], accs[:, 0], accs[:, 1],
                label=labels[0], c='k', ecolor='gray')
ax.axhline(baseline_svd, color=c0, ls='--', label=labels[1])
ax.axhline(baseline_rp, color=c1, ls='--', label=labels[2])
ax.set(xlabel='$n$', #ylim=(0, 6),
       ylabel='distortion (ϵ)',
       title='distortion of reducing \n $N=${:.0e} dims to $k={}$ dims'.format(N, k))
ax.legend()
panel += 1

# timing
ax = plt.subplot(gs_top[1])
ax.text(in1, in2, chr(panel), transform=ax.transAxes, 
                        fontsize=16, fontweight='bold', va='top', ha='right')
h = ax.errorbar(rp_range[1:], times[1:, 0], times[1:, 1], 
                 label=labels[0], c='k', ecolor='gray')
ax.axhline(baseline_svd_time, color=c0, ls='--', label=labels[1])
ax.axhline(baseline_rp_time, color=c1, ls='--', label=labels[2])
ax.set(xlabel='$n$', ylabel='time (ms)', 
        title='time to reduce \n $N=${:.0e} dims to $k={}$ dims'.format(N, k))
ax.legend()
panel += 1

# pareto of both
ax = plt.subplot(gs_top[2])
ax.text(in1, in2, chr(panel), transform=ax.transAxes, 
                        fontsize=16, fontweight='bold', va='top', ha='right')
im = ax.scatter(times[:, 0], accs[:, 0], c=rp_range, 
                    cmap=cmap, norm=mcolors.LogNorm())
ax.set(xlabel='time (ms)', ylabel='distortion (ϵ)',
        title='pareto front of \n distortion vs timing')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('$n$', rotation=270)
panel += 1

# ---------------------- BOTTOM TWO ROWS --------------------- #
in1, in2 = -0.06, 1.1 # slightly lower panel labels

# setting up ax ndarray from gridspec
ax = np.zeros((2,3), dtype='O')
ax_mid_share = fig.add_subplot(gs_bottom[0, 0])  # picking random to share axes
ax_bot_share = fig.add_subplot(gs_bottom[1, 0])
for i in range(3):
    for j in range(2):
        if j == 0:
            ax[j, i] = fig.add_subplot(gs_bottom[j, i], 
                                       sharex=ax_mid_share, sharey=ax_mid_share)
        else:
            ax[j, i] = fig.add_subplot(gs_bottom[j, i], 
                                       sharex=ax_bot_share, sharey=ax_bot_share)
        ax[j, i].set(xticks=[], yticks=[])
        ax[j, i].text(in1, in2, chr(panel), transform=ax[j, i].transAxes, 
                      fontsize=16, fontweight='bold', va='top', ha='right')
        panel += 1

cmap = cm.get_cmap('Dark2')
ts = np.array([10, 100, 1000]) - l1  # -l1 for init
curr_spikes = spikes[:, 1010:1050]  # random trial
full_projs = [Us[:, :, -1].T.dot(curr_spikes), 
             Qs[:, :, -1].T.dot(curr_spikes)]

plane = 0 # plane 0 is singular plane 1
ind1, ind2 = plane, plane+1
for i, t in enumerate(ts):
    trial_label = t + l1
    for j, bases in enumerate((Us, Qs)):
        basis = bases[:, :, t]
        curr_proj = basis.T @ curr_spikes
        
        # projection onto currently learned subspace
        if j == 0: # have to flip directions sometimes
            ax[j, i].plot(curr_proj[ind1, :], curr_proj[-ind2, :], color=cmap(j))
            ax[j, i].scatter(curr_proj[ind1, 0], curr_proj[-ind2, 0], color=cmap(j))
        else:
            ax[j, i].plot(curr_proj[ind1, :], curr_proj[ind2, :], color=cmap(j))
            ax[j, i].scatter(curr_proj[ind1, 0], curr_proj[ind2, 0], color=cmap(j))

        # projection onto final subspace
        full_proj = full_projs[j]
        ax[j, i].plot(full_proj[ind1, :], full_proj[ind2, :], c='k', alpha=.5, ls='--')
        ax[j, i].scatter(full_proj[ind1, 0], full_proj[ind2, 0], c='k')
    
    ax[1, i].set(xlabel='ssSVD basis vector 1', ylabel='ssSVD basis vector 2')
    ax[0, i].set(title='{} bins ({} s) seen'.format(trial_label, trial_label * (bin_size/1000)),
                 xlabel='singular vector 1', ylabel='singular vector 2')

custom_lines = [Line2D([0], [0], color=cmap(0)),
                Line2D([0], [0], color='k', ls='--')]
ax[0, 2].legend(custom_lines, ['streaming SVD', 'whole data SVD'], loc='upper left')

custom_lines = [Line2D([0], [0], color=cmap(1)),
                Line2D([0], [0], color='k', ls='--')]
ax[1, 2].legend(custom_lines, ['ssSVD', 'whole data ssSVD'], loc='lower right')

title = 'Projection of a single trial onto singular vectors (top) or ssSVD basis vectors (bottom)'
ax[0,1].text(2, 1.28, title, va='top', ha='right', fontsize=14,
             transform=ax[0,1].transAxes)
# fig.savefig('neurips/fig1.pdf', bbox_inches='tight')


