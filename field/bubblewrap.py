import numpy
import networkx as nx
import jax.numpy as np
from math import floor
import time
from collections import deque

from field.utils import center_mass

from jax import jit, grad, vmap, value_and_grad
import jax.scipy.stats
from jax.scipy.stats import multivariate_normal as jmvn
from scipy.stats import multivariate_normal as mvn
from jax.scipy.special import logsumexp as lse

import matplotlib.tri as mtri
from jax import nn, random
from jax.ops import index, index_update
from jax.experimental import optimizers
from jax._src.lax.linalg import triangular_solve


epsilon = 1e-10

class Bubblewrap():
    def __init__(self, num, dim, seed=42, M=30, step=1e-6, lam=1, eps=3e-2, nu=1e-2, B_thresh=1e-4, n_thresh=5e-4, t_wait=1, batch=False, batch_size=1, go_fast = False):
        self.N = num            # Number of nodes
        self.d = dim            # dimension of the space
        self.seed = seed
        self.lam_0 = lam
        self.nu = nu
        
        self.eps = eps
        self.B_thresh = B_thresh
        self.n_thresh = n_thresh
        self.t_wait = t_wait
        self.step = step

        self.batch = batch
        self.batch_size = batch_size
        if not self.batch: self.batch_size = 1

        self.printing = False

        self.go_fast = go_fast
        
        self.key = random.PRNGKey(self.seed)
        numpy.random.seed(self.seed)

        # observations of the data; M is how many to keep in history
        if self.batch: M=self.batch_size
        self.obs = Observations(self.d, M=M, go_fast=go_fast)
        self.get_mus0 = jit(vmap(get_mus, 0))
        self.mu_orig = None
        
    def init_nodes(self):
        ### Based on observed data so far of length M
        
        self.mu = np.zeros((self.N, self.d))

        com = center_mass(self.mu)
        if len(self.obs.saved_obs) > 1:
            obs_com = center_mass(self.obs.saved_obs)
        else:
            ## this section for if we init nodes with no data
            obs_com = 0
            self.obs.curr = com
            self.obs.obs_com = com

        self.mu += obs_com

        prior = (1/self.N)*np.ones(self.N)

        self.alpha = self.lam_0 * prior
        self.last_alpha = self.alpha.copy()
        self.lam = self.lam_0 * prior 
        self.n_obs = 0*self.alpha

        self.mu_orig = self.mu.copy() 
        self.mus_orig = self.get_mus0(self.mu_orig) 

        ### Initialize model parameters (A,En,...)
        self.A = np.ones((self.N,self.N)) - np.eye(self.N)
        self.A /= np.sum(self.A, axis=1)
        self.B = np.zeros((self.N))
        self.En = np.zeros((self.N,self.N)) 

        self.S1 = np.zeros((self.N,self.d))
        self.S2 = np.zeros((self.N,self.d,self.d))

        self.log_A = np.zeros((self.N,self.N))

        fullSigma = numpy.zeros((self.N,self.d,self.d), dtype="float32")
        self.L = numpy.zeros((self.N,self.d,self.d))
        self.L_diag = numpy.zeros((self.N,self.d))
        if self.batch and not self.go_fast:
            var = self.obs.cov
        else:
            var = np.diag(np.var(np.array(self.obs.saved_obs), axis=0))
        for n in numpy.arange(self.N):
            fullSigma[n] = var * (self.nu + self.d + 1) / (self.N**(2/self.d))
            
            ## Optimization is done with L split into L_lower and L_diag elements
            ## L is defined using cholesky of precision matrix, NOT covariance
            L = np.linalg.cholesky(fullSigma[n])
            self.L[n] = np.linalg.inv(L).T
            self.L_diag[n] = np.log(np.diag(self.L[n]))        
        self.L_lower = np.tril(self.L,-1)        
        self.sigma_orig = fullSigma[0] 

        ## Set up gradients
        ## Change grad to value_and_grad if we want Q values
        self.grad_all = jit(vmap(jit(grad(Q_j, argnums=(0,1,2,3))), in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,0)))

        ## Other jitted functions
        self.logB_jax = jit(vmap(single_logB, in_axes=(None, 0, 0, 0)))
        self.expB_jax = jit(expB)
        self.update_internal_jax = jit(update_internal)
        self.kill_nodes = jit(kill_dead_nodes)
        self.log_pred_prob = jit(log_pred_prob)
        self.sum_me = jit(sum_me)
        self.compute_L = jit(vmap(get_L, (0,0)))
        self.get_amax = jit(amax)

        ## for adam gradients
        self.m_mu = np.zeros_like(self.mu)
        self.m_L = np.zeros_like(self.L_lower)
        self.m_L_diag = np.zeros_like(self.L_diag)
        self.m_A = np.zeros_like(self.A)

        self.v_mu = np.zeros_like(self.mu)
        self.v_L = np.zeros_like(self.L_lower)
        self.v_L_diag = np.zeros_like(self.L_diag)
        self.v_A = np.zeros_like(self.A)

        ## Variables for keeping track of dead nodes
        self.dead_nodes = np.arange(0,self.N).tolist()
        self.dead_nodes_ind = self.n_thresh*numpy.ones(self.N)
        self.current_node = 0 
    
        ## Variables for tracking progress
        self.pred = []
        self.teleported_times = []
        self.time_em = []
        self.time_observe = []
        self.time_updates = []
        self.time_grad_Q = []
        self.time_pred = []
        self.entropy_list = []
        ## If we use value_and_grad above to look at Q over time
        self.Q_list = []

        self.t = 1


    def observe(self, x):
        # Get new data point and update observation history

        ## Do all observations, and then update mu0, sigma0
        if self.batch:
            for o in x: # x array of obsevations
                self.obs.new_obs(o)
        # timer = time.time()
        else:
            self.obs.new_obs(x)

        
        if not self.go_fast and self.obs.cov is not None and self.mu_orig is not None:
            lamr = 0.02
            eta = np.sqrt(lamr * np.diag(self.obs.cov))
            
            self.mu_orig = (1-lamr)*self.mu_orig + lamr*self.obs.mean + eta*numpy.random.normal(size=(self.N, self.d))
            self.sigma_orig = self.obs.cov * (self.nu + self.d + 1) / (self.N**(2/self.d))
         

    def em_step(self):
        # take step in E and M; after observation
            
        if self.batch:
            for o in self.obs.saved_obs:
                self.single_em_step(o)
        else:
            self.single_em_step(self.obs.curr)


    def single_em_step(self, x):

        self.beta = 1 + 10/(self.t+1)

        self.B = self.logB_jax(x, self.mu, self.L, self.L_diag)
        
        ### Compute log predictive probability and entropy; turn off for faster code 
        if not self.go_fast:
            new_log_pred = self.log_pred_prob(self.B, self.A, self.alpha) 
            self.pred.append(new_log_pred)
            ent = entropy(self.A, self.alpha)
            self.entropy_list.append(ent)

        ## timing
        timer = time.time()
        new_log_pred = self.log_pred_prob(self.B, self.A, self.alpha) 
        self.time_pred.append(time.time()-timer)

        self.update_B(x)

        self.gamma, self.alpha, self.En, self.S1, self.S2, self.n_obs = self.update_internal_jax(self.A, self.B, self.alpha, self.En, self.eps, self.S1, x, self.S2, self.n_obs)
        
        self.t += 1     


    def update_B(self, x):
        
        if np.max(self.B) < self.B_thresh:
            if not (self.dead_nodes):
                target = numpy.argmin(self.n_obs)

                if self.printing:
                    print('-------------- killing a node: ', target)

                n_obs = numpy.array(self.n_obs)
                n_obs[target] = 0
                self.n_obs = n_obs

                self.remove_dead_nodes()

            node = self.teleport_node(x)
            self.B = self.logB_jax(x, self.mu, self.L, self.L_diag)

        self.current_node, self.B = self.expB_jax(self.B)


    def remove_dead_nodes(self):

        ma = (self.n_obs + self.dead_nodes_ind) < self.n_thresh

        if ma.any():
            ind2 = self.get_amax(ma) 
        
            # try:
            self.n_obs, self.S1, self.S2, self.En, self.log_A = self.kill_nodes(ind2, self.n_thresh, self.n_obs, self.S1, self.S2, self.En, self.log_A)
            actual_ind = int(ind2)
            self.dead_nodes.append(actual_ind)
            self.dead_nodes_ind[actual_ind] = self.n_thresh
            if self.printing:
                print('Removed dead node ', actual_ind, ' at time ', self.t)
            

    def teleport_node(self, x):
        node = self.dead_nodes.pop(0)
        
        mu = numpy.array(self.mu)
        mu[node] = x
        self.mu = mu

        alpha = numpy.array(self.alpha)
        alpha[node] = 1        
        self.alpha = alpha

        self.dead_nodes_ind[node] = 0

        if self.printing:
            print('Teleported node ', node, ' to current data location at time ', self.t)
            self.teleported_times.append(self.t)

        return node


    def grad_Q(self):

        divisor = 1+self.sum_me(self.En)
        (grad_mu, grad_L, grad_L_diag, grad_A) = self.grad_all(self.mu, self.L_lower, self.L_diag, self.log_A, self.S1, self.lam, self.S2, self.n_obs, self.En, self.nu, self.sigma_orig, self.beta, self.d, self.mu_orig)

        self.run_adam(grad_mu/divisor, grad_L/divisor, grad_L_diag/divisor, grad_A/divisor)
        
        self.A = sm(self.log_A)

        self.L = self.compute_L(self.L_diag, self.L_lower)

        # self.Q_list.append(Q_value)


    def run_adam(self, mu, L, L_diag, A):
        ## inputs are gradients
        self.m_mu, self.v_mu, self.mu = single_adam(self.step, self.m_mu, self.v_mu, mu, self.t, self.mu)
        self.m_L, self.v_L, self.L_lower = single_adam(self.step, self.m_L, self.v_L, L, self.t, self.L_lower)
        self.m_L_diag, self.v_L_diag, self.L_diag = single_adam(self.step, self.m_L_diag, self.v_L_diag, L_diag, self.t, self.L_diag)
        self.m_A, self.v_A, self.log_A = single_adam(self.step, self.m_A, self.v_A, A, self.t, self.log_A)


beta1 = 0.99
beta2 = 0.999

### A ton of jitted functions for fast code execution

@jit
def single_adam(step, m, v, grad, t, val):
    m = beta1*m + (1-beta1)*grad
    v = beta2*v + (1-beta2)*grad**2
    m_hat = m/(1-np.power(beta1,t+1))
    v_hat = v/(1-np.power(beta2,t+1))
    update = step*m_hat / (np.sqrt(v_hat)+epsilon)
    val -= update
    return m, v, val
        
@jit
def sm(log_A):
    return nn.softmax(log_A, axis=1)

@jit
def sum_me(En):
    return np.sum(En)

@jit
def amax(A):
    return np.argmax(A)

@jit
def get_L(x, y):
    return np.tril(np.diag(np.exp(x) + epsilon) + np.tril(y,-1))
@jit
def get_L_inv(L):
    return np.linalg.inv(L)

@jit
def get_sig_inv(L):
    return L @ L.T

@jit
def get_fullSigma(L):
    inv = np.linalg.inv(L)
    return inv.T @ inv

@jit
def get_sub_l(L):
    return L.flatten()/np.linalg.norm(L.flatten())

@jit
def get_mus(mu):
    return np.outer(mu,mu)

@jit
def get_ld(L):
    return -2 * np.sum(L)

@jit
def Q_j(mu, L_lower, L_diag, log_A, S1, lam, S2, n_obs, En, nu, sigma_orig, beta, d, mu_orig):
    L = np.tril(np.diag(np.exp(L_diag) + epsilon) + np.tril(L_lower,-1))
    sig_inv = L @ L.T
    mus = np.outer(mu,mu)
    mus_orig = np.outer(mu_orig,mu_orig)
    ld = -2 * np.sum(L_diag)
    
    summed = 0
    summed += (S1 + lam * mu_orig).dot(sig_inv).dot(mu) 
    summed += (-1/2) * np.trace( (sigma_orig + S2 + lam * mus_orig + (lam + n_obs) * mus) @ sig_inv ) 
    summed += (-1/2) * (nu + n_obs + d + 2) * ld
    summed += np.sum((En + beta - 1) * nn.log_softmax(log_A)) 
    return -np.sum(summed)

@jit
def single_logB(x, mu, L, L_diag):
    n = mu.shape[0]
    B = (-1/2) * np.linalg.norm((x-mu)@L)**2  - (n/2) * np.log(2*np.pi) + np.sum(L_diag)
    return B

@jit
def expB(B):
    max_Bind = np.argmax(B)
    current_node = max_Bind
    B -= B[max_Bind]
    B = np.exp(B)
    return current_node, B

@jit
def update_internal(A, B, last_alpha, En, eps, S1, obs_curr, S2, n_obs):
    gamma = B * A / (last_alpha.dot(A).dot(B) + 1e-16)
    alpha = last_alpha.dot(gamma)
    En = gamma * last_alpha[:,np.newaxis] + (1-eps) * En
    S1 = (1 - eps)*S1 + alpha[:,np.newaxis] * obs_curr
    S2 = (1 - eps)*S2 + alpha[:,np.newaxis,np.newaxis] * (obs_curr[:,np.newaxis] * obs_curr.T)
    n_obs = (1 - eps)*n_obs + alpha
    return gamma, alpha, En, S1, S2, n_obs

@jit
def kill_dead_nodes(ind2, n_thresh, n_obs, S1, S2, En, log_A):
    N = n_obs.shape[0]
    d = S1.shape[1]
    # for n_i in ind2:
    n_obs = index_update(n_obs, index[ind2], 0)
    S1 = index_update(S1, index[ind2], np.zeros(d))
    S2 = index_update(S2, index[ind2], np.zeros((d,d)))
    log_A = index_update(log_A, index[ind2], np.zeros(N))
    log_A = index_update(log_A, index[:,ind2], np.zeros(N))
    return n_obs, S1, S2, En, log_A

@jit
def log_pred_prob(B, A, alpha):
    return np.log(alpha @ A @ np.exp(B) + 1e-16)

@jit
def entropy(A, alpha):vmap(get_L, (0,0))(self.L_diag, self.L_lower)
       

        self.curr = None 
        self.saved_obs = deque(maxlen=self.M)

        self.mean = None
        self.last_mean = None

        self.cov = None
        
        self.n_obs = 0

    def new_obs(self, coord_new):
        self.curr = coord_new
        self.saved_obs.append(self.curr)
        self.n_obs += 1

        if not self.go_fast:
            if self.mean is None:
                self.mean = self.curr.copy()
            else:
                self.last_mean = self.mean.copy()
                self.mean = update_mean(self.mean, self.curr, self.n_obs)

            if self.n_obs > 2:
                if self.cov is None:
                    self.cov = np.cov(np.array(self.saved_obs).T, bias=True)
                else:
                    self.cov = update_cov(self.cov, self.last_mean, self.curr, self.mean, self.n_obs)


@jit 
def update_mean(mean, curr, n_obs):
    return mean + (curr - mean)/n_obs

@jit
def update_cov(cov, last, curr, mean, n):
    lastm = get_mus(last)
    currm = get_mus(mean)
    curro = get_mus(curr)
    f = (n-1)/n
    return f*(cov + lastm) + (1-f)*curro - currm