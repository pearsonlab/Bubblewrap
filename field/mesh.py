from collections import deque
from itertools import islice

import networkx as nx
import numpy as np
from networkx.drawing.layout import spring_layout

import field.mesh_jax as mj
from field.utils import bounding, center_mass, dumb_bounding

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

## Class Mesh for set of adaptive mesh points to trace flows
## All using linear interpolation in places; could generalize later
## Terminology: mesh points have fixed neighbors, observation points have bounding points
class Mesh:
    def __init__(self, num, dim=3, M=10, **kwargs):
        self.num = num  # num is number of points per dimension
        self.N = num**dim  # total number of mesh points
        self.d = dim  # dimension of the space
        self.spr = 0.5  # spring constant in potential
        self.step = 5e-1
        self.seed = 42

        # coordinates for each mesh point
        self.coords = np.zeros((self.N, self.d), dtype="float32")  
        # magnitude/direction components on each mesh point
        self.vectors = np.zeros((self.N, self.d), dtype="float32")  
        # (variable number of) neighbors for each mesh point
        self.neighbors = [None] * self.N  
        # distance matrix from midpoint observation to mesh points
        self.dist = np.zeros(self.N)  
        # equil. pt for spring dist  #TODO: set this as variable
        self.a = 1  # +np.zeros(self.d)                 

        self.G = nx.Graph()
        self.G.add_nodes_from(np.arange(0,self.N))

        # define initial mesh point spacing; all vectors are 0 magnitude and direction
        # self.initialize_mesh()

        # initially no observations
        self.obs = Observations(self.d, M=M)
        self.pred = None

        ## other useful parameters
        # rotation: (alpha*v1 + beta*v2) / (alpha+beta)
        self.alpha = 1  # scale current mesh vector
        self.beta = 1  # scale observed vector

    def initialize_mesh(self):
        # sets coords and neighbors
        # neighbors: ideally if coords are (N,d) this is (N,k) containing k indices of 1-distance neighbors

        sl = [slice(0, self.num)] * self.d
        self.coords = np.mgrid[sl].reshape((self.d, self.N)).astype("float32").T
        # NOTE: could subtract CoM here
        # TODO: add initial scale to warm up phase

        # for initialization after mesh.. redo order sometime
        com = center_mass(self.coords)
        obs_com = center_mass(self.obs.saved_obs)

        self.coords -= com
        scale = np.max(np.abs(self.obs.saved_obs - obs_com))*2 / self.num
        self.coords *= scale
        self.coords += obs_com
        self.a *= scale

        ##### neighbors
        ## TODO: an actually good implementation of this; see manhatten
        for i in np.arange(0, self.N):
            d = np.linalg.norm(self.coords - self.coords[i], axis=1)
            self.neighbors[i] = np.squeeze(np.argwhere(np.abs(d - scale) < 1e-4))
            self.G.add_edges_from([(i,n) for n in self.neighbors[i]])

        # TODO: decide on initialization for vectors; zero seems less good
        # Currently using random directions, length 1 (very small compared to mesh size atm)
        self.vectors = np.random.default_rng(self.seed).random((self.N, self.d)) - 0.5
        scalev = np.linalg.norm(self.vectors, axis=1) 
        self.vectors = (self.vectors.T / scalev).T * scale

        # for later ease of comparison
        self.coords0 = self.coords.copy()
        self.vectors0 = self.vectors.copy()

    def observe(self, coord_new):
        # update observation history
        self.obs.new_obs(coord_new)
        
        ## TODO: update self.a based on spread of all obs; add spread to obs class

    def predict(self, p=1):
        # given new observed vector and its neighbor, what's our pred for that point
        # p is power for inverse dist weighting, higher for smoother interp
        # TODO: sanity check for blow-ups here, though springs should prevent it
        
        # find new bounding points and their distances to the observed midpoint
        self.dist, self.bounding = bounding(self.coords, self.obs.mid, num=2**self.d)
        
        dists = self.dist[self.bounding]

        if np.any(dists == 0):  # TODO: use min dist not zero, set param elsewhere
            # we have a prediction at this point already, no need to interp
            # bounding is sorted so it's the first one
            self.pred = self.vectors[self.bounding[0]]
            # TODO: check no movement of this one from spatial update

        weights = 1 / (dists**p)
        self.weights = weights
        self.pred = weights.dot(self.vectors[self.bounding]) / np.sum(weights)

    def grad_pred(self):
        # NOTE: assuming p=1 here
        # take gradient of prediction wrt distance
        V = self.pred  # if self.pred has dist==0 then..?
        Z = np.sum(self.weights)

        # need to restrict to active bounding points
        dist = self.dist[self.bounding]
        dwdx = (self.obs.mid - self.coords[self.bounding]) / dist[:, None]**3

        grad = 2*np.abs(V - self.obs.vect) * dwdx * (self.vectors[self.bounding] - V) / Z

        ####
        # print('------original min value', np.sum(np.abs(V-self.obs.vect)**2))

        # TODO: need to also deal with boundary case when prediction is on a mesh point
        self.coords[self.bounding] -= grad * self.step  # (step size)

        dists = np.linalg.norm(self.coords - self.obs.curr, axis=1)  #new distances
        dist = dists[self.bounding]
        grad_vec = 2*(V - self.obs.vect) / (Z*dist[:, None])

        self.vectors[self.bounding] -= grad_vec * self.step

        # ###########
        
        # weights = 1/(dist)      ## new prediciton
        # self.pred = weights.dot(self.vectors[self.bounding])/np.sum(weights)
        # print('------new min value', np.sum(np.abs(V-self.obs.vect)**2))

        self.step /= 1.001

    def jax_grad(self):
        args = [self.coords[self.bounding], self.vectors[self.bounding], self.obs.mid, self.obs.vect]
                
        Δ = grad(mj.loss)(*args)
        self.coords[self.bounding] -= Δ * self.step

        args[2] = self.obs.curr
        Δ_vec = grad(mj.loss, argnums=1)(*args)
        self.vectors[self.bounding] -= Δ_vec * self.step
        
        self.step /= 1.001

    def evaluate(self):
        # for all observations in memory
        pass

    def relax(self):
        # step done after gradient and point movings
        # For each mesh point, get its neighbors and distances
        # TODO: better implementation; can at least group by number of neighbors

        for i in np.arange(0, self.N):
            dists = np.linalg.norm(self.coords[i] - self.coords[self.neighbors[i]][:, None], axis=1)
            # possibly .T; output (N,d,k)
            try:
                dij = np.linalg.norm(dists, axis=1)
                poten = self.spr*(dij - self.a)*(dists.T)/dij

                direc = np.sign(self.coords[i] - self.coords[self.neighbors[i]])

                self.coords[self.neighbors[i]] -= poten.T * direc * self.step*10  # (step_size_here)
            except:
                breakpoint()    #TODO

            newdists = np.linalg.norm(self.coords[i] - self.coords[self.neighbors[i]][:, None], axis=1)
            meand = np.mean(newdists)
            if not meand or meand > 200:
                print("nan here")
                print("mean dist", meand)

                import pdb

                pdb.set_trace()

        # TODO: this needs to change if we're thinking of spatial propagation

    def relax_network(self):
        init_pos = dict((i,c.tolist()) for i,c in enumerate(self.coords))
        fixed_nodes = self.bounding.tolist()
        new_pos = spring_layout(self.G, k=self.a, pos=init_pos, fixed=fixed_nodes, dim=self.d)
        self.coords = np.array([p for p in new_pos.values()], dtype="float32")

    def rotate(self, scaling="global"):
        if scaling == "global":
            # apply same rotational scaling to all points
            self.vectors = (self.alpha*self.vectors + self.beta*self.obs.vect)/(self.alpha + self.beta)

        elif scaling == "dist":
            # apply rotation scaled by distance to observed midpoint
            norm = (self.alpha + self.beta) * self.dist  # CHECK dimension here
            self.vectors = (self.alpha*self.dist @ self.vectors + self.beta*self.dist @ self.obs.vect)/norm

    # def shift_mesh(self):
    #     # for initialization after mesh.. redo order sometime
    #     com = center_mass(self.coords)
    #     self.coords -= com

    #     # for later ease of comparison
    #     self.coords0 = self.coords.copy()

    def grad_step(self):
        # compute gradient and take a step in that direction
        # |obs - pred|^2 + k|d_ij - a|^2
        pass

    def quiet(self, data):
        # add data to observations as initial set
        # don't move mesh in any way (move this? also update flow vecs)
        for i in np.arange(0, data.shape[0]):
            # self.obs.saved_obs.append(data[i])
            self.obs.new_obs(data[i])
        # self.obs.last = self.obs.saved_obs[i]


## Class to store last few observations inside the mesh
## TODO: Maybe split into separate file, if it evolves into a more complex class
class Observations:
    def __init__(self, dim, M=5):
        self.M = M  # how many observed points to hold in memory
        self.d = dim  # dimension of coordinate system

        self.curr = None  # np.zeros(self.d)
        self.last = np.zeros(self.d)
        self.vect = None
        self.mid = None

        self.saved_obs = deque(maxlen=self.M)
        self.mid_list = deque(maxlen=self.M)
        self.vect_list = deque(maxlen=self.M)

    def new_obs(self, coord_new):
        self.curr = coord_new
        self.vect = self.curr - self.last
        self.mid = self.last + 0.5*self.vect

        self.last = coord_new

        # if len(self.saved_obs)==self.M:
        #     self.saved_obs.popleft()
        self.saved_obs.append(self.curr)
        self.mid_list.append(self.mid)
        self.vect_list.append(self.vect)

    def get_last_obs(self, n=1):
        # get the last n observations, n<=self.M
        # returns list in order last obs, last-1 obs, last-2 obs, etc
        ## TODO: this doesn't work
        if n > self.M:
            n = self.M
        self.saved_obs.rotate(-n)
        last = list(islice(self.saved_obs, 0, n, 1)).reverse()
        self.saved_obs.rotate(n)
        return last


if __name__ == "__main__":

    import matplotlib.pylab as plt
    from datagen import plots
    from datagen.models import lorenz, vanderpol
    from scipy.integrate import solve_ivp

    # Define parameters

    T = 1000
    dt = 0.1
    M = 50
    num = 10
    internal_reps = 3

    ## Generate some data; shape (T,dim)
    # 3D lorenz system
    # x0, y0, z0 = (0.1, 0.1, 0.1)
    # dim = 3
    # sln = solve_ivp(lorenz, (0, T), (x0, y0, z0), args=(), dense_output=True, rtol=1e-6)

    # 2d vdp oscillator
    x0, y0 = (0.1, 0.1)
    dim = 2
    sln = solve_ivp(vanderpol, (0, T), (x0, y0), args=(), dense_output=True, rtol=1e-6)
    
    t = np.linspace(0, dt*T, T)
    data = sln.sol(t).T * 100 #scale makes for easier human readability imo
    
    ## Plotting during mesh refinement
    # fig, axs = plt.subplots(ncols=2)


    # Initialize mesh [around data]
    mesh = Mesh(num, dim=dim, M=M)
    # Give first few obs without doing anything
    mesh.quiet(data[:M, :])
    mesh.initialize_mesh()

    for i in np.arange(0, T - M):
        # get new observation
        mesh.observe(data[i+M])
        for j in np.arange(0,internal_reps):
            # get our prediction for that obs
            # fig, axs = plt.subplots(ncols=3)

            mesh.predict()

            # bp = mesh.bounding
            # plots.plot_color(data[:i+M+1, 0], data[:i+M+1, 1], t[:i+M+1], axs[0])
            # axs[0].quiver(mesh.coords[:, 0], mesh.coords[:, 1], mesh.vectors[:, 0], mesh.vectors[:, 1], color='m')
            # axs[0].quiver(mesh.coords[bp, 0], mesh.coords[bp, 1], mesh.vectors[bp, 0], mesh.vectors[bp, 1], color='k')
            # axs[0].quiver(mesh.obs.mid[0], mesh.obs.mid[1], mesh.obs.vect[0], mesh.obs.vect[1], color='r')
            # axs[0].quiver(mesh.obs.mid[0], mesh.obs.mid[1], mesh.pred[0], mesh.pred[1], color='g')

            # axs[1].quiver(mesh.coords[bp, 0], mesh.coords[bp, 1], mesh.vectors[bp, 0], mesh.vectors[bp, 1], color='k')
            # axs[1].quiver(mesh.obs.mid[0], mesh.obs.mid[1], mesh.pred[0], mesh.pred[1], color='g')
            
            # spatial/vector gradient updates
            # mesh.grad_pred()
            mesh.jax_grad()

            # bp = mesh.bounding
            # m = np.sum(mesh.coords0 - mesh.coords, axis=1) != 0
            # v = np.sum(mesh.vectors0 - mesh.vectors, axis=1) != 0

            # # breakpoint()
            # plots.plot_color(data[:i+M+1, 0], data[:i+M+1, 1], t[:i+M+1], axs[1])
            # axs[1].quiver(mesh.obs.mid[0], mesh.obs.mid[1], mesh.obs.vect[0], mesh.obs.vect[1], color='r')
            # axs[1].quiver(mesh.coords[:, 0], mesh.coords[:, 1], mesh.vectors[:, 0], mesh.vectors[:, 1], color='m')
            # axs[1].quiver(mesh.coords[bp, 0], mesh.coords[bp, 1], mesh.vectors[bp, 0], mesh.vectors[bp, 1], color='gray')
            # axs[1].quiver(mesh.obs.mid[0], mesh.obs.mid[1], mesh.pred[0], mesh.pred[1], color='lime')

            # axs[2].quiver(mesh.coords[bp, 0], mesh.coords[bp, 1], mesh.vectors[bp, 0], mesh.vectors[bp, 1], color='g')
            

            # adjust vectors of bounding points
            # mesh.grad_vec()
            # spring relaxation gradient update
            mesh.relax_network()

            # m = np.sum(mesh.coords0 - mesh.coords, axis=1) != 0
            # v = np.sum(mesh.vectors0 - mesh.vectors, axis=1) != 0

            # # breakpoint()
            # # axs[2].quiver(mesh.coords[m, 0], mesh.coords[m, 1], mesh.vectors[m, 0], mesh.vectors[m, 1], color='m')
            # axs[2].quiver(mesh.coords[:, 0], mesh.coords[:, 1], mesh.vectors[:, 0], mesh.vectors[:, 1], color='m')
            # plots.plot_color(data[:i+M+1, 0], data[:i+M+1, 1], t[:i+M+1], axs[2])
            # axs[2].quiver(mesh.coords[bp, 0], mesh.coords[bp, 1], mesh.vectors[bp, 0], mesh.vectors[bp, 1], color='b')
            # axs[2].quiver(mesh.obs.mid[0], mesh.obs.mid[1], mesh.obs.vect[0], mesh.obs.vect[1], color='r')
            
            # yl = axs[2].get_ylim()
            # xl = axs[2].get_xlim()
            # axs[0].set_ylim(yl)
            # axs[0].set_xlim(xl)
            # axs[1].set_ylim(yl)
            # axs[1].set_xlim(xl)

            # plt.show()

    # breakpoint()
    # import matplotlib.pylab as plt

    # mask not moved points
    # m = np.sum(mesh.coords0 - mesh.coords, axis=1) == 0
    # mesh.vectors[m] = np.zeros(dim)

    ###### 3d plot
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')

    # ax.quiver(mesh.coords[:,0], mesh.coords[:,1], mesh.coords[:,2], mesh.vectors[:,0],mesh.vectors[:,1], mesh.vectors[:,2])
    # cmap = plt.cm.plasma
    # for i in range(0,T):
    #     ax.scatter(data[i,0], data[i,1], data[i,2], color=cmap(i/T))

    ###### 2d plots
    # from datagen import plots

    if dim>2:
        fig, axs = plt.subplots(ncols=dim)
        for i in np.arange(dim-1):
            plots.plot_color(data[:, i], data[:, i+1], t, axs[i])
            axs[i].quiver(mesh.coords[:, i], mesh.coords[:, i+1], mesh.vectors[:, i], mesh.vectors[:, i+1])
        plots.plot_color(data[:, 0], data[:, dim-1], t, axs[dim-1])
        axs[dim-1].quiver(mesh.coords[:, 0], mesh.coords[:, dim-1], mesh.vectors[:, 0], mesh.vectors[:, dim-1])

    else: # TODO: this is specific to 2d case, could generalize
        fig, axs = plt.subplots(ncols=2)

        bp = mesh.bounding

        plots.plot_color(data[:, 0], data[:, 1], t, axs[0])
        axs[0].quiver(mesh.coords0[:, 0], mesh.coords0[:, 1], mesh.vectors0[:, 0], mesh.vectors0[:, 1])
        plots.plot_color(data[:, 0], data[:, 1], t, axs[1])
        axs[1].quiver(mesh.coords[:, 0], mesh.coords[:, 1], mesh.vectors[:, 0], mesh.vectors[:, 1], color='k')
        # mid = np.asarray(mesh.obs.mid_list)
        # vect = np.asarray(mesh.obs.vect_list)
        # axs[1].quiver(mid[:, 0], mid[:, 1], vect[:, 0], vect[:, 1])

        axs[0].title.set_text('Initial grid (random)')
        axs[1].title.set_text('Final grid, 1 step/new observation')

    plt.autoscale()
    plt.show()
