# import numpy as np
import numpy
import networkx as nx
import jax.numpy as np
from math import floor
import time

from field.mesh import Observations
from field.utils import center_mass

from jax import jit, grad, vmap, value_and_grad
import jax.scipy.stats
from jax.scipy.stats import multivariate_normal as jmvn
from scipy.stats import multivariate_normal as mvn
from jax.scipy.special import logsumexp as lse

import matplotlib.tri as mtri
# from scipy.special import softmax, log_softmax
from jax import nn, random
from jax.ops import index, index_update
from jax.experimental import optimizers
from jax._src.lax.linalg import triangular_solve


epsilon = 1e-10
# np.set_printoptions(precision=4)
# np.set_printoptions(suppress=True)
# np.seterr(invalid='raise')

# import jax
# jax.config.update('jax_platform_name', 'cpu')
# from jax.config import config
# config.update("jax_debug_nans", True)


## Working title: 'Graph quantized dynamical systems' (gqds) --> need to rename to Bubblewrap TODO

class GQDS():
    def __init__(self, num, num_d, dim, seed=42, M=30, step=1e-6, lam=1, eps=3e-2, nu=1e-2, sigma_scale=1e3, mu_scale=2, kappa=1e-2, n_thresh=1e-6, B_thresh=1e-4, t_wait=200):
        self.N = num            # Number of nodes
        self.d = dim            # dimension of the space
        self.seed = seed
        self.step = step
        self.lam_0 = lam
        self.nu_0 = nu
        self.eps = eps
        self.sigma_scale = sigma_scale
        # self.beta = beta      ## NOTE: beta defined in em ~1/t
        self.kap = kappa
        self.n_thresh = n_thresh
        self.B_thresh = B_thresh
        self.t_wait = t_wait
        self.num_d = num_d
        self.mu_scale = mu_scale
        
        ## TODO: setup proper logging
        self.printing = True
        
        ## not actually used
        self.key = random.PRNGKey(self.seed)
        # np.random.seed(self.seed)

        # observations of the data; M is how many to keep in history
        self.obs = Observations(self.d, M=M)

        self.alpha = (1/self.N)*np.ones(self.N)

        self.time_observe = []
        self.time_updates = []
        self.time_grad_Q = []
        
    def init_nodes(self):
        ### Compute initial ss based on observed data so far
        # set initial centers of nodes distributed across space of observations
        # sl = [slice(0, self.num_d)] * self.d       
        # breakpoint()
        # self.mu = np.array(np.meshgrid(*[np.linspace(0,self.num_d,self.num_d)]*self.d, sparse=False)).reshape((self.d, self.N)).T
        self.mu = np.zeros((self.N, self.d))

        com = center_mass(self.mu)
        if len(self.obs.saved_obs) > 1:
            obs_com = center_mass(self.obs.saved_obs)
            self.scale = np.max(np.abs(self.obs.saved_obs - obs_com))*2  / self.num_d
        else:
            ## this section for if we init mesh with no data
            obs_com = 0
            self.obs.curr = com
            self.obs.obs_com = com
            self.scale = 1

        self.mu -= com
        self.scale *= self.mu_scale
        self.mu *= self.scale
        self.mu += obs_com

        prior = (1/self.N)*np.ones(self.N)

        self.alpha = self.lam_0 * prior
        self.last_alpha = self.alpha.copy()
        self.lam = self.lam_0 * prior 
        self.nu = self.nu_0 * prior

        self.n_obs = 0*self.alpha
        self.mu_orig = self.mu.copy()

        ### Initialize model parameters (A,En,...)
        self.A = np.ones((self.N,self.N)) - np.eye(self.N)#*0.99999
        self.A /= np.sum(self.A, axis=1)
        self.B = np.zeros((self.N))
        self.En = np.zeros((self.N,self.N)) #(1/self.N) * np.ones((self.N,self.N))

        self.S1 = np.zeros((self.N,self.d))
        self.S2 = np.zeros((self.N,self.d,self.d))

        self.mus_orig = vmap(get_mus, 0)(self.mu)

        self.log_A = np.zeros((self.N,self.N))


        self.fullSigma = numpy.zeros((self.N,self.d,self.d), dtype="float32")
        self.L = numpy.zeros((self.N,self.d,self.d))
        self.L_diag = numpy.zeros((self.N,self.d))
        for n in numpy.arange(self.N):
            self.fullSigma[n] = numpy.diagflat(self.sigma_scale*(1/self.scale)*numpy.ones((self.d), dtype="float32"))*(1/self.N) / (self.nu[n] + self.d + 2 +  self.n_obs[n])#[...,None]

        self.fullSigma_orig = self.fullSigma.copy()

        ## this is important
        self.update_ss()       

        for n in np.arange(self.N):
            L = np.linalg.cholesky(self.fullSigma[n])
            self.L[n] = np.linalg.inv(L).T
            self.L_diag[n] = np.log(np.diag(self.L[n]))        
        ### NOTE: L is now defined using cholesky of precision matrix, NOT covariance!
        self.L_lower = np.tril(self.L,-1) #np.zeros((self.N,self.d,self.d)) 

        self.A_orig = self.A.copy()
        
        self.t = 1

        ## Set up gradients
        self.grad_all = jit(value_and_grad(Q_est, argnums=(0,1,2,3)))

        ## other jitted functions
        self.logB_jax = jit(vmap(single_logB, in_axes=(None, 0, 0)))
        self.expB_jax = jit(expB)
        self.update_internal_jax = jit(update_internal)
        self.kill_nodes = jit(kill_dead_nodes)

        ## for adam gradients
        ## TODO: rewrite optimally?
        self.m_mu = np.zeros_like(self.mu)
        self.m_L = np.zeros_like(self.L_lower)
        self.m_L_diag = np.zeros_like(self.L_diag)
        self.m_A = np.zeros_like(self.A)

        self.v_mu = np.zeros_like(self.mu)
        self.v_L = np.zeros_like(self.L_lower)
        self.v_L_diag = np.zeros_like(self.L_diag)
        self.v_A = np.zeros_like(self.A)

        self.dead_nodes = np.arange(0,self.N).tolist()
        self.dead_nodes_ind = self.n_thresh*numpy.ones(self.N)
        self.current_node = 0   #?
    
        ## variables for tracking progress
        self.pred = []
        self.teleported_times = []

        self.Q_list = []
        self.A_diff = []

    # @profile
    def observe(self, x):
        # Get new data point and update observation history
        timer = time.time()
        self.obs.new_obs(x)
        self.time_observe.append(time.time()-timer)

    # @profile
    def log_pred_prob(self):
        return np.log(np.sum(np.exp(self.B) * self.A[self.current_node] * self.alpha[self.current_node]) + 1e-16)

    # @profile
    def em_step(self):
        # take step in E and M; after observation

        timer = time.time()
        self.last_alpha = self.alpha.copy()
        
        self.beta = 1 + 10/(self.t+1)

        # if self.t>self.t_wait:       
        #     self.remove_dead_nodes()

        self.B = self.logB_jax(self.obs.curr, self.mu, self.L)
        # print(self.log_pred_prob())
        # self.pred.append(self.log_pred_prob())

        self.update_B()

        self.gamma, self.alpha, self.En, self.S1, self.S2, self.n_obs = self.update_internal_jax(self.A, self.B, self.alpha, self.last_alpha, self.En, self.eps, self.S1, self.obs.curr, self.S2, self.n_obs)

        self.grad_Q()
        
        self.t += 1        

    # @profile
    def update_B(self):
        
        if np.max(self.B) < self.B_thresh:
            if not (self.dead_nodes):
                ## got to kill a node!
                ### Is this redundant with remove_dead_nodes() logic??      ##FIXME
                target = np.argmin(self.n_obs)
                # self.n_obs[target] = 0
                print('-------------- killing a node: ', target)

                n_obs = numpy.array(self.n_obs)
                n_obs[target] = 0
                self.n_obs = n_obs

                self.remove_dead_nodes()

            ## if we have any free nodes and no node is nearby this datapoint
            node = self.teleport_node()
            ## TODO: faster if we only recompute one part and copy back?
            self.B = self.logB_jax(self.obs.curr, self.mu, self.L)

        self.current_node, self.B = self.expB_jax(self.B)

    def update_ss(self):
        # Update sufficient statistics

        self.mu = (self.lam[:,None] * self.mu_orig + self.S1) / (self.lam + self.n_obs)[:,None]
        mus = vmap(get_mus, 0)(self.mu)
        self.fullSigma = (self.fullSigma_orig + self.lam[:,None,None]*self.mus_orig + self.S2 - (self.lam + self.n_obs)[:,None,None] * mus) / (self.nu + self.d + 1 + self.n_obs)[:,None,None]

    # @profile
    def remove_dead_nodes(self):

        ma = (self.n_obs + self.dead_nodes_ind) < self.n_thresh

        if ma.any():
            ind2 = np.argmax(ma)
        
            # try:
            self.log_A = self.kill_nodes(ind2, self.n_obs, self.n_thresh, self.log_A)
            actual_ind = int(ind2)
            self.dead_nodes.append(actual_ind)
            self.dead_nodes_ind[actual_ind] = self.n_thresh
            print('Removed dead node ', actual_ind, ' at time ', self.t)
            # print(self.dead_nodes)
            # print(self.dead_nodes_ind)
        # except:
        #     breakpoint()
        # #     print('oops')

    #     ## if any nodes n_obs<thresh, remove
    #     aw = numpy.argwhere(self.n_obs < self.n_thresh)
    #     aw = aw.flat
    #     ind = set(aw)
    #     # try:
    #     #     ind2 = [i for i in ind if i not in self.dead_nodes]
    #     # except:
    #     #     breakpoint()
    #     # if ind2:
    #     ind2 = ind - self.dead_nodes
    #     self.dead_nodes.update(ind2)
    #     # s = len(ind2)
    #     # breakpoint()
    #     for n_i in ind2:
    #         self.log_A = index_update(self.log_A, index[n_i], np.zeros(self.N))
    #         self.log_A = index_update(self.log_A, index[:,n_i], np.zeros(self.N))
    #     # update = np.zeros_like(self.log_A)
    #     # a = numpy.array(self.log_A)
    #     # a[tuple(ind2), :] = 0
    #     # a[:, tuple(ind2)] = 0
    #     # self.log_A = a #self.log_A - update
    #     # if self.printing:
    #     #     print('Removed dead nodes: ', ind2)
    #         # print(self.dead_nodes)

    # @profile
    def teleport_node(self):
        node = self.dead_nodes.pop(0)

        # mu_update = (self.lam[node] * self.obs.curr + self.S1[node]) / (self.lam[node] + self.n_obs[node])

        # self.mu[node] = self.obs.curr
        # mu_update = self.obs.curr
        # self.mu = index_update(self.mu, index[node], mu_update) #self.obs.curr)
        
        mu = numpy.array(self.mu)
        mu[node] = self.obs.curr
        self.mu = mu

        alpha = numpy.array(self.alpha)
        alpha[node] = 1        
        self.alpha = alpha

        self.dead_nodes_ind[node] = 0

        if self.printing:
            print('Teleported node ', node, ' to current data location at time ', self.t)
            self.teleported_times.append(self.t)

        return node

    # @profile
    def grad_Q(self):

        Q_value, (grad_mu, grad_L, grad_L_diag, grad_A) = self.grad_all(self.mu, self.L_lower, self.L_diag, self.log_A, self.lam, self.S1, self.S2, self.En, self.nu, self.n_obs, self.beta*np.ones((self.N,1)), self.mu_orig, self.fullSigma_orig, self.d*np.ones((self.N,1)), self.mus_orig, self.obs.obs_com)
        
        ## adam
        self.run_adam(grad_mu, grad_L, grad_L_diag, grad_A)
        
        self.A = sm(self.log_A)

        self.L = vmap(get_L, (0,0))(self.L_diag, self.L_lower)
       
        self.Q_list.append(Q_value)

        self.step /= 1.001      # ?

    # @profile
    def run_adam(self, mu, L, L_diag, A):
        ## inputs are gradients
        self.m_mu, self.v_mu, self.mu = single_adam(self.step, self.m_mu, self.v_mu, mu, self.t, self.mu)
        self.m_L, self.v_L, self.L_lower = single_adam(self.step, self.m_L, self.v_L, L, self.t, self.L_lower)
        self.m_L_diag, self.v_L_diag, self.L_diag = single_adam(self.step, self.m_L_diag, self.v_L_diag, L_diag, self.t, self.L_diag)
        self.m_A, self.v_A, self.log_A = single_adam(self.step, self.m_A, self.v_A, A, self.t, self.log_A)

beta1 = 0.99
beta2 = 0.999

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
def Q_est(mu, L, L_diag, log_A, lam, S1, S2, En, nu, n_obs, beta, mu_orig, sigma_orig, d, mus_orig, com):

    N = log_A.shape[0]
    d = mu.shape[1]
    t = 1+np.sum(En)

    ## is this even faster? yes
    el = vmap(get_L, (0,0))(L_diag,L)
    sig_inv = vmap(get_sig_inv, 0)(el)
    mus = vmap(get_mus, 0)(mu)
    ld = vmap(get_ld, 0)(L_diag)

    # summed = 0
    # for j in jnp.arange(N):

    #     # ld = -2 * jnp.sum(L_diag[j])
    #     # mus = jnp.outer(mu[j], mu[j])

    #     summed += (S1[j] ).dot(sig_inv[j]).dot(mu[j]) 
    #     summed += (-1/2) * jnp.trace( (sigma_orig[j] + S2[j] + (n_obs[j]) * mus[j]) @ sig_inv[j] ) 
    #     summed += (-1/2) * (nu[j] + n_obs[j] + d + 2) * ld[j]
    #     summed += jnp.sum((En[j] + beta - 1) * nn.log_softmax(log_A[j])) 

    summed = vmap(Q_j, 0)(S1, lam, sig_inv, mu, mu_orig, sigma_orig, S2, n_obs, mus, mus_orig, nu, ld, En, log_A, beta)

    return -np.sum(summed)/t 

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
def Q_j(S1, lam, sig_inv, mu, mu_orig, sigma_orig, S2, n_obs, mus, mus_orig, nu, ld, En, log_A, beta):
    summed = 0
    summed += (S1 + lam * mu_orig).dot(sig_inv).dot(mu) 
    summed += (-1/2) * np.trace( (sigma_orig + S2 + lam * mus_orig + (lam + n_obs) * mus) @ sig_inv ) 
    summed += (-1/2) * (nu + n_obs + 3 + 2) * ld
    summed += np.sum((En + beta - 1) * nn.log_softmax(log_A)) 
    return summed

## already explicitly jit-ing these..?
def single_logB(x, mu, L):
    # inv = np.linalg.inv(L)
    # fullSigma = inv.T @ inv
    # B = jmvn.logpdf(x, mu, fullSigma)

    ## from jax github to improve logpdf of mvn
    n = mu.shape[0]
    el = np.linalg.inv(L).T
    y = triangular_solve(el, x-mu, lower=True, transpose_a=True)
    B = (-1/2) * np.einsum('...i,...i->...', y, y) - (n/2) * np.log(2*np.pi) - np.log(el.diagonal()).sum()
    return B

def expB(B):
    max_Bind = np.argmax(B)
    current_node = max_Bind
    B -= B[max_Bind]
    B = np.exp(B)
    return current_node, B


def update_internal(A, B, alpha, last_alpha, En, eps, S1, obs_curr, S2, n_obs):
    gamma = B * A / (alpha.dot(A).dot(B))
    alpha = last_alpha.dot(gamma)
    En = gamma * last_alpha[:,np.newaxis] + (1-eps) * En
    S1 = (1 - eps)*S1 + alpha[:,np.newaxis] * obs_curr
    S2 = (1 - eps)*S2 + alpha[:,np.newaxis,np.newaxis] * (obs_curr[:,np.newaxis] * obs_curr.T)
    n_obs = (1 - eps)*n_obs + alpha
    return gamma, alpha, En, S1, S2, n_obs


def kill_dead_nodes(ind2, n_obs, n_thresh, log_A):
    N = n_obs.shape[0]
    # for n_i in ind2:
    log_A = index_update(log_A, index[ind2], np.zeros(N))
    log_A = index_update(log_A, index[:,ind2], np.zeros(N))
    return log_A