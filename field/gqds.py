import numpy as np
import jax.numpy as jnp
from math import floor

from field.mesh import Observations
from field.utils import center_mass

from jax import jit, grad
import jax.scipy.stats
from jax.scipy.stats import multivariate_normal as jmvn
from scipy.stats import multivariate_normal as mvn

epsilon = 1e-16
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


## Working title: 'Graph quantized dynamical systems' (gqds)

class GQDS():
    def __init__(self, num, dim, seed=42, M=10, max_child=10, step=1e-6, lam=1, eps=1e-4, nu=1e-2, sigma_scale=1e6):
        self.N = num            # Number of nodes
        self.d = dim            # dimension of the space
        self.seed = seed
        self.max_child = max_child
        # self.step = step
        self.lam_0 = lam
        self.nu_0 = nu
        # self.eps = eps
        self.sigma_scale = sigma_scale

        # parameters for Gaussian distributions 
        self.mu = np.zeros((self.N,self.d), dtype="float32")
        self.sigma = np.ones((self.N), dtype="float32")

        # observations of the data; M is how many to keep in history
        self.obs = Observations(self.d, M=M)

        # HMM variables
        
        # self.Q = 0
        # self.gamma = None
        self.alpha = np.zeros((self.N))
        self.En = (1/self.N) * np.ones((self.N,self.N))

        # self.S1 = 0
        # self.S2 = 0
        
    def init_nodes(self):
        ### Compute initial ss based on observed data so far
        # set initial centers of nodes distributed across space of observations
        sl = [slice(0, floor(np.sqrt(self.N)))] * self.d       
        self.mu = np.mgrid[sl].reshape((self.d, self.N)).astype("float32").T

        com = center_mass(self.mu)
        obs_com = center_mass(self.obs.saved_obs)

        self.mu -= com
        scale = np.max(np.abs(self.obs.saved_obs - obs_com))*2  / self.d / floor(np.sqrt(self.N))
        scale *= 15
        self.mu *= scale
        self.mu += obs_com

        self.fullSigma = np.zeros((self.N,self.d,self.d), dtype="float32")
        for n in np.arange(self.N):
            self.fullSigma[n] = np.diagflat(self.sigma_scale*(1/scale)*np.ones((self.d), dtype="float32"))   

        ### Initialize model parameters (A,En,...)
        prior = (1/self.N)*np.ones(self.N)

        self.alpha = self.lam_0 * prior
        self.last_alpha = self.alpha
        self.lam = self.lam_0 * prior 
        self.nu = self.nu_0 * prior

        self.n_obs = 0

        self.A = np.ones((self.N,self.N)) - np.eye(self.N)#*0.99999
        self.A /= np.sum(self.A, axis=1)
        self.B = np.zeros((self.N))
        self.En = (1/self.N) * np.ones((self.N,self.N))

        self.S1 = np.zeros((self.N,self.d))
        self.S2 = np.zeros((self.N,self.d,self.d))

        self.fullSigma *= prior[:,None,None]
        self.mus_orig = np.zeros((self.N,self.d,self.d))
        for n in np.arange(self.N):
            self.mus_orig[n] = np.outer(self.mu[n], self.mu[n])

        ### Save copies of mu, sigma, A for later comparison
        self.mu_orig = self.mu.copy()
        self.fullSigma_orig = self.fullSigma.copy()
        self.A_orig = self.A.copy()

        ## With these new values, update ss
        self.t = 1
        self.update_ss()

    def observe(self, x):
        # Get new data point and update observation history
        self.obs.new_obs(x)

        self.last_alpha = self.alpha

    def em_step(self):
        # take step in E and M; after observation

        # print(self.obs.curr)
        # print(self.mu)
        print(self.alpha)

        self.update_B()
        self.update_gamma()
        self.update_alpha()
        self.update_En()
        self.update_A()

        self.update_S()
        self.update_ss()

        self.t += 1

    def update_B(self):
        # Compute posterior over graph nodes (?)
        for n in np.arange(self.N):
            try:
                self.B[n] = mvn.logpdf(self.obs.curr, mean=self.mu[n], cov=self.fullSigma[n])
            except:
                breakpoint()
        
        max_Bind = np.argmax(self.B)
        self.B -= self.B[max_Bind]
        self.B = np.exp(self.B)

        print(self.t)
        # breakpoint()
        # print(self.B)
        # self.B[self.B<epsilon] = epsilon

    def update_gamma(self):
        # Compute new update matrix gamma
        # gamma_{l,h}(T) = B_{h,x_T} A_{l,h} / sum_{i,j}(B_{j,x_T} A_{i,j} alpha_i(T-1))

        self.gamma = self.B * self.A / (self.alpha.dot(self.A).dot(self.B)) # + epsilon)

    
    def update_alpha(self):
        # Compute new posterior marginal probabiltiies alpha
        # alpha_j(T) = sum_i (gamma_{i,j}(T) alpha_i(T-1))

        self.alpha = self.last_alpha.dot(self.gamma)
        self.alpha[self.alpha<epsilon] = epsilon
        self.alpha /= np.sum(self.alpha)
        

    def update_En(self):
        # Compute new N matrix [N_{i,j} = sum_t p(z_t=j, z_{t+1} = 1 | x_{1:T})]
        # N_{i,j} = (1/T) gamma_{i,j}(T) alpha_i(T) + (1-1/T) N_{i,j}(T-1)

        self.En = (1/self.t) * self.gamma * self.alpha[:,np.newaxis] + (1-1/self.t) * self.En
        

    def update_A(self):
        # Compute updated state evolution matrix A using N
        # A_{i,j} = N_{i,j} / sum_j N_{i,j}

        self.A = self.En / (np.sum(self.En, axis=1)[:,np.newaxis] + epsilon)


    def update_S(self):
        # update S tensor
        # S1(T+1) = (1-eps)*S1(T) + eps*alpha(T) @ x
        # S2 sim but @ x xT
        # S1 shape: (N,d)
        # S2 shape: (N,d,d)

        self.S1 += self.alpha[:,np.newaxis] * self.obs.curr
        self.S2 += self.alpha[:,np.newaxis,np.newaxis] * (self.obs.curr[:,np.newaxis] * self.obs.curr.T)
        self.n_obs += self.alpha
        
    
    def update_ss(self):
        # Update sufficient statistics

        self.mu = (self.lam[:,None] * self.mu_orig + self.S1) / (self.lam + self.n_obs)[:,None]
        mus = np.zeros_like(self.S2)
        for n in np.arange(self.N):
            mus[n] = np.outer(self.mu[n], self.mu[n])
        self.fullSigma = (self.fullSigma_orig + self.lam[:,None,None]*self.mus_orig + self.S2 - (self.lam + self.n_obs)[:,None,None] * mus) / (self.nu + self.d + 1 + self.n_obs)[:,None,None]
