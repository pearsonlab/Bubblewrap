import numpy as np
import matplotlib.pylab as plt
from scipy.integrate import solve_ivp

## Class Mesh for set of adaptive mesh points to trace flows
## All using linear interpolation in places; could generalize later
## Terminology: mesh points have fixed neighbors, observation points have bounding points
class Mesh():

    def __init__(self, num, dim, **kwargs):
        self.N = num        # number of mesh points
        self.d = dim        # dimension of the space
        self.spr = 1        # spring constant in potential

        self.coords = [np.zeros(self.d)]*self.N     # coordinates for each mesh point
        self.vectors = [np.zeros(self.d)]*self.N    # mangitude/direction components on each mesh point
        self.dist = np.zeros(self.N)                # distance matrix from midpoint observation to mesh points
        self.a = 1+np.zeros(self.d)                 # equil. pt for spring dist

        # define initial mesh point spacing; all vectors are 0 magnitude and direction
        # TODO: initial update of vectors from 0 different than subsquent ones?

        #TODO: static datastructure indexed by mesh point ID as above. Could initialize via callout method.
        self.neighbors = None   
        # ideally if coords are (N,d) this is (N,d,k) for k-nearest neighbors? or (N,k,d)
        
        
        # initially no observations
        # TODO: separate class?
        self.o_curr = None #np.zeros(self.d)
        self.o_last = None
        self.o_vect = None
        self.o_mid = None

        self.pred = None
        
        ## other useful parameters
        # rotation: (alpha*v1 + beta*v2) / (alpha+beta)
        self.alpha = 1      # scale current mesh vector
        self.beta = 1       # scale observed vector

    def observe(self, coord_new):
        self.o_curr = coord_new
        self.o_vect = self.o_curr - self.o_last
        self.o_mid = self.o_curr + 0.5*self.o_vect

        self.dist, self.bounding = bounding(self.o_mid, self.coords, num=4)

        # self.o_last = self.o_curr

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
            self.vectors = (self.alpha*self.vectors + self.beta*self.o_vect)/(self.alpha+self.beta)

        elif scaling=='dist':
            # apply rotation scaled by distance to observed midpoint
            norm = (self.alpha+self.beta)*self.dist #CHECK dimension here
            self.vectors = (self.alpha*(self.dist)@self.vectors + self.beta*(self.dist)@self.o_vect)/norm

    def shift(self):
        # use dist scaling, global would be ridiculous
        pass

    def evaluate(self):
        # effectively evaluate the 'monitor' function on the grid to equipartition uncertainty ?
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


def bounding(ref, points, num=4):
    # choose num nearest bounding points on the mesh (e.g. 4, assuming roughly square 2D grid)
    # of ref given set of points, that is the new observation

    dist = np.linalg.norm(points - ref)             #TODO: check performance vs math.dist, scipy euclidean, etc
    bounding = np.argsort(dist)[:num]              #TODO: same note, check argpartition

    # NOTE: need a method for bounding, not just nearest, points

    return dist, bounding
