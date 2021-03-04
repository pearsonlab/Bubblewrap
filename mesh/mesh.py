import numpy as np
from collections import deque
from itertools import islice
from scipy.optimize import linprog
from scipy.spatial import Delaunay

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
        self.spr = 1  # spring constant in potential
        self.step = 1

        # coordinates for each mesh point
        self.coords = np.zeros((self.N, self.d), dtype="float32")  
        # magnitude/direction components on each mesh point
        self.vectors = np.zeros((self.N, self.d), dtype="float32")  
        # (variable number of) neighbors for each mesh point
        self.neighbors = [None] * self.N  
        # distance matrix from midpoint observation to mesh points
        self.dist = np.zeros(self.N)  
        # equil. pt for spring dist  #TODO: set this as variable
        self.a = 10  # +np.zeros(self.d)                 

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

        # for later ease of comparison
        self.coords0 = self.coords.copy()

        ##### neighbors
        ## TODO: an actually good implementation of this; see manhatten
        for i in np.arange(0, self.N):
            d = np.linalg.norm(self.coords - self.coords[i], axis=1)
            self.neighbors[i] = np.squeeze(np.argwhere(np.abs(d - scale) < 1e-4))

        # TODO: decide on initialization for vectors; zero seems less good
        # Currently using random directions, length 1 (very small compared to mesh size atm)
        self.vectors = np.random.random((self.N, self.d)) - 0.5
        scale = np.linalg.norm(self.vectors, axis=1)
        self.vectors = (self.vectors.T / scale).T

    def observe(self, coord_new):
        # update observation history
        self.obs.new_obs(coord_new)
        

    def predict(self, p=1):
        # given new observed vector and its neighbor, what's our pred for that point
        # p is power for inverse dist weighting, higher for smoother interp
        # TODO: sanity check for blow-ups here, though springs should prevent it
        
        # find new bounding points and their distances to the observed midpoint
        self.dist, self.bounding = bounding(self.coords, self.obs.mid)
        
        dists = self.dist[self.bounding]

        if np.any(dists == 0):  # TODO: use min dist not zero, set param elsewhere
            # we have a prediction at this point already, no need to interp
            # bounding is sorted so it's the first one
            self.pred = self.vectors[self.bounding[0]]

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

        # TODO: closed-form
        # grad = np.zeros((self.bounding.shape[0], self.d))
        # for i,b in np.ndenumerate(self.bounding):
        #     grad[i] = -(self.vectors[b].T)/(Z*dist[i]**2) + V.T/(Z*dist[i]**2)
        # # grad = -(self.vectors[self.bounding].T)/(Z*dist**2) + V.T/(Z*dist**2)
        # import pdb; pdb.set_trace()

        dwdx = (self.obs.mid - self.coords[self.bounding]) / dist[:, None]**3

        grad = 2*(V - self.obs.vect) * dwdx * (self.vectors[self.bounding] - V) / Z
        # np.abs? no effect?

        ####
        # print('------original min value', np.sum(np.abs(V-self.obs.vect)**2))

        # TODO: need to also deal with boundary case when prediction is on a mesh point
        self.coords[self.bounding] -= grad * self.step  # (step size)

        grad_vec = 2*(V - self.obs.vect) / (Z*dist[:, None])

        self.vectors[self.bounding] -= grad_vec * self.step

        # ###########
        # dists = np.linalg.norm(self.coords - self.obs.curr, axis=1)  #new distances
        # dist = dists[self.bounding]
        # weights = 1/(dist)      ## new prediciton
        # V = weights.dot(self.vectors[self.bounding])/np.sum(weights)
        # print('------new min value', np.sum(np.abs(V-self.obs.vect)**2))

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

                self.coords[self.neighbors[i]] += poten.T * direc * self.step/5  # (step_size_here)
            except:
                import pdb

                pdb.set_trace()

            newdists = np.linalg.norm(self.coords[i] - self.coords[self.neighbors[i]][:, None], axis=1)
            meand = np.mean(newdists)
            if not meand or meand > 200:
                print("nan here")
                print("mean dist", meand)

                import pdb

                pdb.set_trace()

        # TODO: this needs to change if we're thinking of spatial propagation

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
            self.obs.saved_obs.append(data[i])

        self.obs.last = self.obs.saved_obs[i]


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

    def new_obs(self, coord_new):
        self.curr = coord_new
        self.vect = self.curr - self.last
        self.mid = self.curr + 0.5*self.vect

        # if len(self.saved_obs)==self.M:
        #     self.saved_obs.popleft()
        self.saved_obs.append(self.curr)

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


def center_mass(points):
    # Compute center of mass of points, assuming equal masses here
    # points is a list of coords arrays [array((dim,))]*N
    # TODO: use average(..., weight=mass_array) in future; 
    # can weight by e.g. similarity or flow vectors

    return np.mean(points, axis=0)


def check_bounded(points, x):
    # From https://stackoverflow.com/a/43564754.
    n_points = len(points)
    c = np.zeros(n_points)
    A = np.r_[points.T, np.ones((1, n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

def dumb_bounding(points, ref, num=8):
    dist = np.linalg.norm(points - ref, axis=1) 
    bounding = np.argsort(dist)[:num]
    return dist, bounding

def bounding(points, ref, num=8):
    # choose num nearest bounding points on the mesh (e.g. 4, assuming roughly square 2D grid)
    # of ref given set of points, that is the new observation
    dirs = points - ref
    zero = ref - ref
    
    # TODO: Use some tree data structure, r-tree, etc.
    dist = np.linalg.norm(dirs, axis=1)
    closest = np.argsort(dist)
    
    if not check_bounded(dirs[closest[:num]], zero):
        # Add closest points one by one until enclosed.
        k = 1
        hull = Delaunay(dirs[closest[: num + k]])
        while (simp := hull.find_simplex(zero)) == -1:  # not bounded
            hull = Delaunay(dirs[closest[: num + k]])
            # hull.add_points(dirs[np.newaxis, closest[idx], :])  # TODO: Need to deal with non-uniqueness.
            k += 1

        # Add simplex vertices, then next closest points.
        important = closest[hull.simplices[simp]]
        bounding = np.zeros(num, dtype=int)
        bounding[: important.size] = important
        k = 0
        for i in range(important.size, num):
            while closest[k] in important:
                k += 1
            bounding[i] = closest[k]
            k += 1

    else:
        bounding = closest[:num]

    # assert check_bounded(points, ref)

    return dist, bounding


if __name__ == "__main__":

    from datagen.models import lorenz
    from scipy.integrate import solve_ivp

    # Define parameters
    np.random.seed(42)

    T = 500
    dt = 0.1
    x0, y0, z0 = (0.1, 0.1, 0.1)
    dim = 3
    M = 10
    num = 10

    # Generate some data; shape (T,dim)
    sln = solve_ivp(lorenz, (0, T), (x0, y0, z0), args=(), dense_output=True, rtol=1e-6)
    t = np.linspace(0, dt*T, T)
    data = sln.sol(t).T

    # data = np.zeros((T,dim))
    # data[:,0] = sln["x"]
    # data[:,1] = sln["y"]
    # data[:,2] = sln["z"]

    # Initialize mesh [around data]
    mesh = Mesh(num, dim=dim, M=M)
    # Give first few obs without doing anything
    mesh.quiet(data[:M, :])
    mesh.initialize_mesh()

    for i in np.arange(0, T - M):
        # get new observation
        mesh.observe(data[i+M])
        for j in np.arange(0,3):
            # get our prediction for that obs
            mesh.predict()
            # spatial gradient update
            mesh.grad_pred()
            # adjust vectors of bounding points
            # mesh.grad_vec()
            # spring relaxation gradient update
            mesh.relax()

    import matplotlib.pylab as plt

    # mask not moved points
    m = np.sum(mesh.coords0 - mesh.coords, axis=1) == 0
    # import pdb; pdb.set_trace()
    mesh.vectors[m] = np.zeros(dim)

    ###### 3d plot
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')

    # ax.quiver(mesh.coords[:,0], mesh.coords[:,1], mesh.coords[:,2], mesh.vectors[:,0],mesh.vectors[:,1], mesh.vectors[:,2])
    # cmap = plt.cm.plasma
    # for i in range(0,T):
    #     ax.scatter(data[i,0], data[i,1], data[i,2], color=cmap(i/T))

    ###### 2d plots
    from datagen import plots

    fig, axs = plt.subplots(ncols=3)
    plots.plot_color(data[:, 0], data[:, 1], t, axs[0])
    axs[0].quiver(mesh.coords[:, 0], mesh.coords[:, 1], mesh.vectors[:, 0], mesh.vectors[:, 1])
    plots.plot_color(data[:, 0], data[:, 2], t, axs[1])
    axs[1].quiver(mesh.coords[:, 0], mesh.coords[:, 2], mesh.vectors[:, 0], mesh.vectors[:, 2])
    plots.plot_color(data[:, 1], data[:, 2], t, axs[2])
    axs[2].quiver(mesh.coords[:, 1], mesh.coords[:, 2], mesh.vectors[:, 1], mesh.vectors[:, 2])

    plt.show()
