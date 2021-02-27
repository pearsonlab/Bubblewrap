import numpy as np
from collections import deque
from itertools import islice

## Class Mesh for set of adaptive mesh points to trace flows
## All using linear interpolation in places; could generalize later
## Terminology: mesh points have fixed neighbors, observation points have bounding points
class Mesh():

    def __init__(self, num, dim=3, **kwargs):
        # num is number of points per dimension
        self.N = num*dim        # total number of mesh points
        self.d = dim            # dimension of the space
        self.spr = 1            # spring constant in potential

        self.coords = np.zeros(self.N,self.d)     # coordinates for each mesh point
        self.vectors = np.zeros(self.N,self.d)    # mangitude/direction components on each mesh point
        self.neighbors = [None]*self.N              # need to account for variable number of neighbors
        self.dist = np.zeros(self.N)                # distance matrix from midpoint observation to mesh points
        self.a = 1+np.zeros(self.d)                 # equil. pt for spring dist

        # define initial mesh point spacing; all vectors are 0 magnitude and direction
        self.initialize_mesh()
        
        # initially no observations
        self.obs = Observations(self.d, M=10)
        self.pred = None
        
        ## other useful parameters
        # rotation: (alpha*v1 + beta*v2) / (alpha+beta)
        self.alpha = 1      # scale current mesh vector
        self.beta = 1       # scale observed vector

    def initialize_mesh(self):
        # sets coords and neighbors 
        # neighbors: ideally if coords are (N,d) this is (N,k) containing k indices of 1-distance neighbors
        
        num = self.N/self.d
        sl = [slice(0,num)]*self.d
        self.coords = np.mgrid[sl].reshape((self.d, self.N)).T
        # NOTE: could subtract CoM here

        ## TODO: an actually good implementation of this; see manhatten
        for i in np.arange(0,self.N):
            d = np.linalg.norm(self.coords-self.coords[i], axis=1)
            self.neighbors[i] = np.squeeze(np.argwhere(d==1))

    def observe(self, coord_new):
        # update observation history
        self.obs.new_obs(coord_new)
        # find new bounding points and their distances to the observed midpoint
        self.dist, self.bounding = bounding(self.obs.mid, self.coords)

    def predict(self, p=1):
        # given new observed vector and its neighbor, what's our pred for that point
        # p is power for inverse dist weighting, higher for smoother interp
        # TODO: sanity check for blow-ups here, though springs should prevent it
        
        dists = self.dist[self.bounding]

        if np.any(dists==0): #TODO: use min dist not zero, set param elsewhere
            # we have a prediction at this point already, no need to interp
            # bounding is sorted so it's the first one
            return self.vectors[self.bounding[0]]

        weights = 1/(dists**p)
        self.weights = weights
        return weights.dot(self.vectors[self.bounding])/np.sum(weights)

    def grad_pred(self):
        # NOTE: assuming p=1 here
        # take gradient of prediction wrt distance
        V =  self.pred              # if self.pred has dist==0 then..?
        Z = np.sum(self.weights)

        # need to restrict to active bounding points
        grad = -self.vectors/(Z*self.dist**2) + V/(Z*self.dist**2)
        return grad

    def rotate(self, scaling='global'):
        if scaling=='global':
            # apply same rotational scaling to all points
            self.vectors = (self.alpha*self.vectors + self.beta*self.obs.vect)/(self.alpha+self.beta)

        elif scaling=='dist':
            # apply rotation scaled by distance to observed midpoint
            norm = (self.alpha+self.beta)*self.dist #CHECK dimension here
            self.vectors = (self.alpha*(self.dist)@self.vectors + self.beta*(self.dist)@self.obs.vect)/norm

    def shift(self):
        # use dist scaling, global would be ridiculous
        pass

    def grad_step(self):
        # compute gradient and take a step in that direction
        # |obs - pred|^2 + k|d_ij - a|^2 
        pass

    def grad_step_vec(self):
        # same, but for vectors, not mesh point locations
        pass

    def relax(self): 
        # step done after gradient and point movings
        # For each mesh point, get its neighbors and distances
        dists = np.linalg.norm(self.coords - self.neighbors) #possibly .T; output (N,d,k)
        poten = self.spr*(dists-self.a)

        self.coords += poten * 1 #(step_size_here)


## Class to store last few observations inside the mesh
## TODO: Maybe split into separate file, if it evolves into a more complex class
class Observations():
    def __init__(self, dim, M=5):
        self.M = M          # how many observed points to hold in memory
        self.d = dim        # dimension of coordinate system

        self.curr = None #np.zeros(self.d)
        self.last = None
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
        if n>self.M:
            n=self.M
        self.saved_obs.rotate(-n)
        last = list(islice(self.saved_obs,0,n,1)).reverse()
        self.saved_obs.rotate(n)
        return last


def center_mass(self, points):
    # Compute center of mass of points, assuming equal masses here
    # points is a list of coords arrays [array((dim,))]*N
    # TODO: use average(..., weight=mass_array) in future; can weight by e.g. similarity or flow vectors

    return np.mean(points, axis=0) 


def bounding(ref, points, num=4):
    # choose num nearest bounding points on the mesh (e.g. 4, assuming roughly square 2D grid)
    # of ref given set of points, that is the new observation

    dist = np.linalg.norm(points - ref)             #TODO: check performance vs math.dist, scipy euclidean, etc
    bounding = np.argsort(dist)[:num]              #TODO: same note, check argpartition

    # NOTE: need a method for bounding, not just nearest, points

    return dist, bounding
