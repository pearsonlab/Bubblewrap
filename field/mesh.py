import time
import networkx as nx
import numpy as np
import jax.numpy as jnp
import matplotlib.tri as mtri
from jax import grad, jit
from networkx.drawing.layout import spring_layout
from collections import deque
from itertools import islice

import field.mesh_jax as mj
from field.utils import bounding, center_mass, dumb_bounding

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

## Class Mesh for set of adaptive mesh points to trace flows
## All using linear interpolation in places; could generalize later
## Terminology: mesh points have fixed neighbors, observation points have bounding points
class Mesh:
    def __init__(self, num, dim=3, M=10, spr=1, step=5e-1, seed=42, neighb=8, mesh_type='square', **kwargs):
        self.num = num  # num is number of points per dimension
        self.N = num**dim  # total number of mesh points
        self.d = dim  # dimension of the space
        self.spr = spr  # spring constant in potential
        self.step = step
        self.seed = seed
        self.max_neighbor = neighb
        self.mesh_type = mesh_type
        # accepted types: 'square', 'triangle'

        # coordinates for each mesh point
        self.coords = np.zeros((self.N, self.d), dtype="float32")  
        # magnitude/direction components on each mesh point
        self.vectors = np.zeros((self.N, self.d), dtype="float32")  
        # (variable number of) neighbors for each mesh point
        self.neighbors = -1 * np.ones((self.N, self.max_neighbor), dtype=int)
        self.n_neighbor = -1 * np.ones(self.N, dtype=int)
        # self.neighbors = [None] * self.N
        # distance matrix from midpoint observation to mesh points
        self.dist = np.zeros(self.N)  
        # equil. pt for spring dist  #TODO: set this as variable

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
        
        # self.grad_coords = jit(grad(mj.prediction_loss))
        # self.grad_vectors = jit(grad(mj.prediction_loss, argnums=1))
        # self.grad_a = jit(grad(mj.spring_energy, argnums=2))

        self.grad_coords = jit(grad(mj.total_loss, argnums=0))
        self.grad_vectors = jit(grad(mj.total_loss, argnums=2))
        self.grad_a = jit(grad(mj.total_loss, argnums=5))

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
        scale *= 5
        self.coords *= scale
        self.coords += obs_com

        ##### neighbors
        ## TODO: an actually good implementation of this; see manhatten
        if self.mesh_type == 'square':
            for i in np.arange(0, self.N):
                d = np.linalg.norm(self.coords - self.coords[i], axis=1)
                nbr = np.squeeze(np.argwhere(np.abs(d - scale) < 1e-4))
                self.n_neighbor[i] = nbr.size
                self.neighbors[i, :nbr.size] = nbr
                self.G.add_edges_from([(i,n) for n in self.neighbors[i, :nbr.size]])
        elif self.mesh_type == 'triangle':
            # NOTE: 2d only here
            self.n_neighbor = np.zeros(self.N, dtype=int)
            triag = mtri.Triangulation(self.coords[:,0], self.coords[:,1])
            for _,n in enumerate(triag.edges):
                a, b = n    #single edge b/t 2 points
                self.neighbors[a, self.n_neighbor[a]] = b
                self.neighbors[b, self.n_neighbor[b]] = a
                self.n_neighbor[a] += 1
                self.n_neighbor[b] += 1
            self.G.add_edges_from(triag.edges)
        else:
            raise Exception

        self.a = self.neighbors.copy()    
        self.a[self.a!=-1] = 1     
        self.a = np.asarray(self.a, dtype='float32')  
        self.a *= scale
        print('Starting a average', np.mean(self.a))


        # TODO: decide on initialization for vectors; zero seems less good
        # Currently using random directions, length 1 (very small compared to mesh size atm)
        self.vectors = np.random.default_rng(self.seed).random((self.N, self.d)) - 0.5
        scalev = np.linalg.norm(self.vectors, axis=1) 
        self.vectors = (self.vectors.T / scalev).T * scale / 10
        
        # Transfer arrays to JAX.
        self.neighbors = jnp.array(self.neighbors)
        self.coords = jnp.array(self.coords)
        self.vectors = jnp.array(self.vectors)

        # for later ease of comparison
        self.coords0 = self.coords.copy()
        self.vectors0 = self.vectors.copy()

    def observe(self, coord_new):
        # update observation history
        self.obs.new_obs(coord_new)
        
    def jax_grad(self):
        args = [self.coords, self.vectors, list(self.obs.mid_list), self.obs.vect]
        self.coords -= self.grad_coords(*args) * self.step #* 50

        args[2] = self.obs.curr
        self.vectors -= self.grad_vectors(*args) * self.step * 0.1

        # arr = self.coords[self.neighbors]  # (points × n_neighbors × dim)
        # mask = (self.neighbors > -1).astype(np.float32)[..., np.newaxis]

        # centered = (arr - self.coords[:, np.newaxis, :]) * mask + 1e-10 # Zero out mask and prevent sqrt(-0).
        # mask2 = np.squeeze(self.a*(1 - mask))
        # ℓ = np.squeeze(np.linalg.norm(centered, axis=2)) + mask2

        # breakpoint()
        
        self.step /= 1.001

    def jax_relax(self):
        # self.a = np.mean(self.obs.scale_list)/self.num * 1.5
        # if new_a > self.a: self.a=new_a
        Δ = grad(mj.spring_energy)(self.coords, self.neighbors, self.a, self.spr)
        self.coords -= np.array(Δ) * self.step * 0.05

        args = [self.coords, self.neighbors, self.a, self.spr]
        self.a -= self.grad_a(*args) * self.step * 2
        # print('New a average', np.mean(self.a))

    def full_grad(self):
        # mask = (self.neighbors > -1).astype(np.float32)[..., np.newaxis]
        # v = self.vectors[self.neighbors]
        # dotprod = 0
        # for i,w in enumerate(v):
        #     dotprod += (self.vectors[i]@w.T)@mask[i]
        #     breakpoint()

        args = [self.coords, self.neighbors, self.vectors, list(self.obs.mid_list), self.obs.vect, self.a, self.spr]
        c_grad = self.grad_coords(*args) * self.step * 1e0
        v_grad = self.grad_vectors(*args) * self.step * 0.1
        a_grad = self.grad_a(*args) * self.step #* 1e-2

        self.coords -= c_grad
        self.vectors -= v_grad
        self.a -= a_grad

        # print('a grad ', np.mean(a_grad))
        # print('a mean ', np.mean(self.a))

        # print('mean coords grad: ', np.mean(c_grad))
        # print('mean vectors grad: ', np.mean(v_grad))

        self.step /= 1.001


    # def relax_network(self):
    #     self.a = np.mean(self.obs.scale_list)/self.num/2
    #     # print('self.a spring length: ', self.a)
    #     init_pos = dict((i,c.tolist()) for i,c in enumerate(self.coords))
    #     fixed_nodes = self.bounding.tolist()
    #     new_pos = spring_layout(self.G, k=self.a, pos=init_pos, fixed=fixed_nodes, dim=self.d)
    #     self.coords = np.array([p for p in new_pos.values()], dtype="float32")

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
        self.obs_com = None

        self.saved_obs = deque(maxlen=self.M)
        self.mid_list = deque(maxlen=self.M)
        self.vect_list = deque(maxlen=self.M)
        self.com_list = deque(maxlen=self.M)
        self.scale_list = deque(maxlen=self.M)

        self.n_obs = 0


    def new_obs(self, coord_new):
        self.curr = coord_new
        self.vect = self.curr - self.last
        self.mid = self.last + 0.5*self.vect

        self.last = coord_new
        self.n_obs += 1

        # if len(self.saved_obs)==self.M:
        #     self.saved_obs.popleft()
        self.saved_obs.append(self.curr)
        self.mid_list.append(self.mid)
        self.vect_list.append(self.vect)

        #TODO: might want to make this over all time, or disallow shrinking?
        if self.obs_com is None:
            self.obs_com = center_mass(self.saved_obs)
        else:
            self.obs_com = self.obs_com + (self.curr - self.obs_com)/self.n_obs
        self.scale = np.max(np.abs(self.saved_obs - self.obs_com))*2
        self.com_list.append(self.obs_com)
        self.scale_list.append(self.scale)

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

