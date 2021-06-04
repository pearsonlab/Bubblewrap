import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
from math import atan2

def plot3d_color(y, t, cmap="viridis", **kwargs):
    """
    Plot a 3D trajectory `y` with colors from `t`.

    Args:
        t (np.ndarray): 1D data.
        y (np.ndarray): n × 3 data.

    Returns:
        Fig, ax
    """
    y = y.T.reshape(-1, 1, 3)

    segments = np.concatenate([y[:-1], y[1:]], axis=1)

    norm = plt.Normalize(t.min(), t.max())
    lc = Line3DCollection(segments, cmap=cmap, norm=norm, alpha=0.9, **kwargs)
    lc.set_array(t)
    lc.set_linewidth(2)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    line = ax.add_collection3d(lc)

    ax.axes.set_xlim3d(
        left=y[:, :, 0].min(), right=y[:, :, 0].max()
    )  # TODO: Find a better way.
    ax.axes.set_ylim3d(bottom=y[:, :, 1].min(), top=y[:, :, 1].max())
    ax.axes.set_zlim3d(bottom=y[:, :, 2].min(), top=y[:, :, 2].max())
    return fig, ax


def plot_color(x, y, t, ax=None, colorbar=True, cmap="viridis", alpha=0.9, **kwargs):
    """
    Plot a trajectory `x` and `y` with colors from `t`.
    """
    y = np.vstack([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([y[:-1], y[1:]], axis=1)

    norm = plt.Normalize(t.min(), t.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=alpha, **kwargs)
    lc.set_array(t)
    lc.set_linewidth(2)

    if ax is None:
        fig, ax = plt.subplots()
    line = ax.add_collection(lc)
    ax.autoscale()
    if colorbar and ax is None:
        fig.colorbar(line)
    return ax


# draws connected graph of gq nodes
def draw_graph(gq, ax, A=None, mu=None, thresh=0.005, alpha=.7, zord=0, node_shape='o', node_color='gray', cmap=plt.cm.binary_r):
    if A is None:
        A = np.array(gq.A)

    if mu is None:
        mu = gq.mu

    G = nx.Graph(A)
    # filter out all edges above threshold and grab id's
    threshold = thresh
    small_edges = list(filter(lambda e: e[2] < threshold, (e for e in G.edges.data('weight'))))
    se_ids = list(e[:2] for e in small_edges)
    print(len(se_ids))
    # remove filtered edges from graph G
    G.remove_edges_from(se_ids)
    G.remove_nodes_from(list(nx.isolates(G))) # get rid of nodes with no edges
    
    # draw
    edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())
    nx.draw(G, mu[:, :2], node_color=node_color, node_size=50, ax=ax,
        edgelist=edges, edge_color=np.asarray(weights), width=2.0, node_shape=node_shape,
        edge_cmap=cmap, linewidths=1, alpha=alpha)
    ax.collections[0].set_edgecolor("#000000")
    return


# scatter of datapoints connected by lines of same colormap
def plot_scatter_connected(data, ax, alpha=0.4):
    xy = data[:, :2].reshape(-1, 1, 2) # taking first 2 cols
    xs, ys = xy[:, 0, :].T
    
    segments = np.hstack([xy[:-1], xy[1:]])
    coll = LineCollection(segments, cmap=plt.cm.viridis, alpha=alpha)
    coll.set_array(np.arange(xy.shape[0]))

    ax.scatter(xs, ys, c=np.arange(xy.shape[0]), alpha=alpha, marker='o')
    ax.add_collection(coll)
    return


# draws gq nodes and variance (if specific nodes, make sure node_list is unique)
def draw_bubbles(gq, sig_ell=1, alpha_ell=.1, ax=None, node_list=None, alpha_node=1, plot_dead_nodes=False):
    if ax is None:
        fig, ax = plt.subplots()

    if node_list is None:
        node_list = np.arange(gq.N)

    for n in node_list:
        if ~plot_dead_nodes and (n in gq.dead_nodes):
        ## don't plot dead nodes
            pass
        else:
            el = np.linalg.inv(gq.L[n][:2,:2])
            sig = el.T @ el
            u,s,v = np.linalg.svd(sig)
            width, height = np.sqrt(s[0])*sig_ell, np.sqrt(s[1])*sig_ell #*=4
            if width>1e5 or height>1e5:
                pass
            else:
                angle = atan2(v[0,1],v[0,0])*360 / (2*np.pi)
                # breakpoint()
                el = Ellipse((gq.mu[n,0],gq.mu[n,1]), width, height, angle, zorder=8)
                el.set_alpha(alpha_ell)
                el.set_clip_box(ax.bbox)
                el.set_facecolor('#FF4400')  ##ed6713')
                ax.add_artist(el)
        
            # plt.text(gq.mu[n,0]+1, gq.mu[n,1], str(n))
            ax.scatter(gq.mu[n,0], gq.mu[n,1], c='k' , zorder=10, alpha=alpha_node)
    return

# plot for log pred prob and entropy
def plot_pred_entropy(gq, ax):
    pred = np.array(gq.pred)
    entropy = np.array(gq.entropy_list)

    ax.plot(pred, 'b', alpha=0.3)
    var_tmp = ewma(pred, 100)
    ax.plot(var_tmp, 'k', lw=1, zorder=10)

    ax2 = ax.twinx()
    ax2.plot(entropy, 'g', alpha=0.3)
    ax2.hlines(np.log2(gq.N), 0, entropy.shape[0], 'g', '--', lw=1, zorder=10)
    var_tmp = ewma(entropy, 100)
    ax2.plot(var_tmp, 'k')
    return ax2

# smoothing for log pred prob and entropy
def ewma(data, com):
    return np.array(pd.DataFrame(data=dict(data=data)).ewm(com).mean()['data'])
