import numpy as np
import networkx as nx
import jax.numpy as jnp
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
from scipy.special import softmax, log_softmax
from jax import nn
from jax.ops import index, index_update

epsilon = 1e-10
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
np.seterr(invalid='raise')

# import jax
# jax.config.update('jax_platform_name', 'cpu')
# from jax.config import config
# config.update("jax_debug_nans", True)


## Working title: 'Graph quantized dynamical systems' (gqds) --> need to rename to LCB (?)

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
        
        np.random.seed(self.seed)

        # observations of the data; M is how many to keep in history
        self.obs = Observations(self.d, M=M)
        ## TODO: redefine M perhaps

        self.alpha = (1/self.N)*np.ones(self.N)

        self.time_observe = []
        self.time_updates = []
        self.time_grad_Q = []
        
    def init_nodes(self):
        ### Compute initial ss based on observed data so far
        # set initial centers of nodes distributed across space of observations
        sl = [slice(0, self.num_d)] * self.d       
        self.mu = np.mgrid[sl].reshape((self.d, self.N)).astype("float32").T

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

        self.mus_orig = np.zeros((self.N,self.d,self.d))
        for n in np.arange(self.N):
            self.mus_orig[n] = np.outer(self.mu[n], self.mu[n])

        self.log_A = np.zeros((self.N,self.N))


        self.fullSigma = np.zeros((self.N,self.d,self.d), dtype="float32")
        self.L = np.zeros((self.N,self.d,self.d))
        self.L_diag = np.zeros((self.N,self.d))
        for n in np.arange(self.N):
            self.fullSigma[n] = np.diagflat(self.sigma_scale*(1/self.scale)*np.ones((self.d), dtype="float32"))*(1/self.N) / (self.nu[n] + self.d + 2 +  self.n_obs[n])#[...,None]

        self.fullSigma_orig = self.fullSigma.copy()
        self.update_ss()       
        
        for n in np.arange(self.N):
            L = np.linalg.cholesky(self.fullSigma[n])
            self.L[n] = np.linalg.inv(L).T
            self.L_diag[n] = np.log(np.diag(self.L[n]))        
        ### NOTE: L is now defined using cholesky of precision matrix, NOT covariance!
        self.L_lower = np.tril(self.L,-1) #np.zeros((self.N,self.d,self.d)) 

        ### Save copies of mu, sigma, A for later comparison
        
        self.A_orig = self.A.copy()

        ## With these new values, update ss
        self.t = 1

        ## Set up gradients
        # self.grad_mu = jit(grad(Q_est, argnums=0))
        # self.grad_L_lower = jit(grad(Q_est, argnums=1))
        # self.grad_L_diag = jit(grad(Q_est, argnums=2))
        # self.grad_A = jit(grad(Q_est, argnums=3))
        self.grad_all = jit(value_and_grad(Q_est, argnums=(0,1,2,3)))

        self.Q_list = []
        self.A_diff = []

        ## for adam gradients
        ## TODO: rewrite optimally
        self.beta1 = np.float32(0.9)
        self.beta2 = np.float32(0.999)

        self.m_mu = np.zeros_like(self.mu)
        self.m_L = np.zeros_like(self.L_lower)
        self.m_L_diag = np.zeros_like(self.L_diag)
        self.m_A = np.zeros_like(self.A)

        self.v_mu = np.zeros_like(self.mu)
        self.v_L = np.zeros_like(self.L_lower)
        self.v_L_diag = np.zeros_like(self.L_diag)
        self.v_A = np.zeros_like(self.A)

        # self.set_neighbors()

        self.dead_nodes = [] 

    def set_neighbors(self):
        self.G = nx.Graph()
        self.G.add_nodes_from(np.arange(0,self.N))
        self.neighbors = np.zeros((self.N, self.N))

        triag = mtri.Triangulation(self.mu[:,0], self.mu[:,1])
        for _,n in enumerate(triag.edges):
            a, b = n    #single edge b/t 2 points
            self.neighbors[a, b] = self.kap
            self.neighbors[b, a] = self.kap
        self.G.add_edges_from(triag.edges)

        ### Adjust by number of neighbors?
        # self.neighbors = self.neighbors / (np.sum(self.neighbors, axis=1)[:,np.newaxis])
        
    # @profile
    def observe(self, x):
        # Get new data point and update observation history
        timer = time.time()
        self.obs.new_obs(x)
        self.time_observe.append(time.time()-timer)

    # @profile
    def em_step(self):
        # take step in E and M; after observation

        timer = time.time()
        self.last_alpha = self.alpha.copy()
        
        self.beta = 1 + 10/(self.t+1)

        if self.t>self.t_wait:       
            self.remove_dead_nodes()

        self.update_B()
        self.update_gamma()
        self.update_alpha()
        self.update_En()
        self.update_S()
        self.time_updates.append(time.time()-timer)

        timer = time.time()
        self.grad_Q()
        self.time_grad_Q.append(time.time()-timer)

        self.t += 1

    # @profile
    def update_B(self):
        # Compute posterior over graph
        # timer = time.time()
        for n in np.arange(self.N):
            inv = np.linalg.inv(self.L[n])
            self.fullSigma[n] = inv.T @ inv
        self.B = multiple_logpdfs(self.obs.curr, self.mu, self.fullSigma)
        # print('new B ', time.time()-timer)

        # if self.printing:
        #     print(np.max(self.B))

        if np.max(self.B) < self.B_thresh:
            if not (self.dead_nodes):
                ## got to kill a node!
                target = np.argmin(self.n_obs)
                self.n_obs[target] = 0
                self.remove_dead_nodes()
            ## if we have any free nodes and no node is nearby this datapoint
            node = self.teleport_node()
            ## recompute B? in other function
            self.B[node] = mvn.logpdf(self.obs.curr, self.mu[node], self.fullSigma[node])

        max_Bind = np.argmax(self.B)
        self.B -= self.B[max_Bind]
        self.B = np.exp(self.B)
        # print('final B ', time.time()-timer)

    def update_gamma(self):
        # Compute new update matrix gamma
        # gamma_{l,h}(T) = B_{h,x_T} A_{l,h} / sum_{i,j}(B_{j,x_T} A_{i,j} alpha_i(T-1))

        self.gamma = self.B * self.A / (self.alpha.dot(self.A).dot(self.B)) # + epsilon)

    
    def update_alpha(self):
        # Compute new posterior marginal probabiltiies alpha
        # alpha_j(T) = sum_i (gamma_{i,j}(T) alpha_i(T-1))

        self.alpha = self.last_alpha.dot(self.gamma)
        # self.alpha[self.alpha<epsilon] = epsilon
        # self.alpha /= np.sum(self.alpha)
        

    def update_En(self):
        # Compute new N matrix [N_{i,j} = sum_t p(z_t=j, z_{t+1} = 1 | x_{1:T})]
        # N_{i,j} = (1/T) gamma_{i,j}(T) alpha_i(T) + (1-1/T) N_{i,j}(T-1)
        
        self.En = self.gamma * self.last_alpha[:,np.newaxis] + (1-self.eps) * self.En
        

    # def update_A(self):
    #     # Compute updated state evolution matrix A using N
    #     # A_{i,j} = N_{i,j} / sum_j N_{i,j}

    #     self.A = self.En / (np.sum(self.En, axis=1)[:,np.newaxis])


    def update_S(self):
        # update S tensor
        # S1(T+1) = (1-eps)*S1(T) + eps*alpha(T) @ x
        # S2 sim but @ x xT
        # S1 shape: (N,d)
        # S2 shape: (N,d,d)

        # self.eps = 1/(self.t+1)

        self.S1 = (1 - self.eps)*self.S1 + self.alpha[:,np.newaxis] * self.obs.curr
        self.S2 = (1 - self.eps)*self.S2 + self.alpha[:,np.newaxis,np.newaxis] * (self.obs.curr[:,np.newaxis] * self.obs.curr.T)
        self.n_obs = (1 - self.eps)*self.n_obs + self.alpha
        
    
    def update_ss(self):
        # Update sufficient statistics

        self.mu = (self.lam[:,None] * self.mu_orig + self.S1) / (self.lam + self.n_obs)[:,None]
        mus = np.zeros_like(self.S2)
        for n in np.arange(self.N):
            mus[n] = np.outer(self.mu[n], self.mu[n])
        self.fullSigma = (self.fullSigma_orig + self.lam[:,None,None]*self.mus_orig + self.S2 - (self.lam + self.n_obs)[:,None,None] * mus) / (self.nu + self.d + 1 + self.n_obs)[:,None,None]

    # @profile
    def remove_dead_nodes(self):
        ## if any nodes n_obs<thresh, remove
        ind = np.argwhere(self.n_obs < self.n_thresh).flatten().tolist()
        try:
            ind2 = [i for i in ind if i not in self.dead_nodes]
        except:
            breakpoint()
        if ind2:
            self.dead_nodes.extend(ind2)
            s = len(ind2)
            # breakpoint()
            for n_i in ind2:
                self.log_A = index_update(self.log_A, index[n_i], jnp.zeros(self.N))
                self.log_A = index_update(self.log_A, index[:,n_i], jnp.zeros(self.N))
            if self.printing:
                print('Removed dead nodes: ', ind2)
                # print(self.dead_nodes)

    # @profile
    def teleport_node(self):
        node = self.dead_nodes.pop(0)

        # mu_update = (self.lam[node] * self.obs.curr + self.S1[node]) / (self.lam[node] + self.n_obs[node])

        # self.mu[node] = self.obs.curr
        mu_update = self.obs.curr
        self.mu = index_update(self.mu, index[node], mu_update) #self.obs.curr)
        self.alpha[node] = 1        

        # self.S1[node] = (1 - self.eps)*self.S1[node] + self.alpha[node] * self.obs.curr
        # self.S2[node] = (1 - self.eps)*self.S2[node] + self.alpha[node] * (self.obs.curr * self.obs.curr.T)
        # self.n_obs[node] = (1 - self.eps)*self.n_obs[node] + self.alpha[node]
        
        # self.mu[node] = (self.lam[node] * self.obs.curr + self.S1[node]) / (self.lam[node] + self.n_obs[node])
        # sig = (self.fullSigma_orig[node] + self.lam[node]*self.mus_orig[node] + self.S2[node] - (self.lam + self.n_obs)[node] * np.outer(self.mu[node], self.mu[node])) / (self.nu[node] + self.d + 1 + self.n_obs[node])


        if self.printing:
            print('Teleported node ', node, ' to current data location at time ', self.t)

        return node

    # @profile
    def grad_Q(self):

        args = [self.mu, self.L_lower, self.L_diag, self.log_A, self.lam, self.S1, self.S2, self.En, self.nu, self.n_obs, self.beta*np.ones((self.N,1)), self.mu_orig, self.fullSigma_orig, self.d*np.ones((self.N,1)), self.mus_orig, self.obs.obs_com]

        # timer = time.time()
        # grad_mu = self.grad_mu(*args)
        # grad_L = np.array(self.grad_L_lower(*args))
        # grad_L_diag = np.array(self.grad_L_diag(*args))
        # grad_A = self.grad_A(*args)
        Q_value, (grad_mu, grad_L, grad_L_diag, grad_A) = self.grad_all(*args)
        # print('Compute gradients ', time.time()-timer)
        
        if (grad_L_diag > 1e3).any():
            print('Large L_diag gradient')
            # breakpoint()

        ## adam
        # timer = time.time()
        updates = self.run_adam(grad_mu, np.array(grad_L), np.array(grad_L_diag), grad_A)
        # print('Compute adam update ', time.time()-timer)
        self.mu -= updates[0] 
        self.L_lower -= updates[1] 
        self.L_diag -= updates[2] 
        self.log_A -= updates[3] 

        self.A = softmax(self.log_A, axis=1)

        # self.A_diff.append(self.A-self.A_orig)
        
        # timer = time.time()
        self.L = np.zeros((self.N,self.d,self.d)) #= np.diagonal(np.exp(self.L_diag)) + self.L_lower
        for n in np.arange(self.N):
            self.L[n] = np.diag(np.exp(self.L_diag[n])) + self.L_lower[n]
        # print('Reconstruct L ', time.time()-timer)
        
        # timer = time.time()
        # Q = -Q_est(*args)
        # print('Evaulate Q ', time.time()-timer)
        # print(self.t) #, Q)
        self.Q_list.append(Q_value)
        # print('-------------------------------after gradient')

        self.step /= 1.001

    # @profile
    def run_adam(self, mu,L,L_diag,A):
        ## inputs are gradients
        ## TODO: rewrite this

        self.m_mu = self.beta1*self.m_mu + (1-self.beta1)*mu
        self.v_mu = self.beta2*self.v_mu + (1-self.beta2)*mu**2
        m_hat_mu = self.m_mu/(1-np.power(self.beta1,self.t+1))
        v_hat_mu = self.v_mu/(1-np.power(self.beta2,self.t+1))
        update_mu = self.step*m_hat_mu / (np.sqrt(v_hat_mu)+epsilon)

        self.m_L = self.beta1*self.m_L + (1-self.beta1)*L
        self.v_L = self.beta2*self.v_L + (1-self.beta2)*L**2
        m_hat_L = self.m_L/(1-np.power(self.beta1,self.t+1))
        v_hat_L = self.v_L/(1-np.power(self.beta2,self.t+1))
        update_L = self.step*m_hat_L / (np.sqrt(v_hat_L)+epsilon)

        self.m_L_diag = self.beta1*self.m_L_diag + (1-self.beta1)*L_diag
        self.v_L_diag = self.beta2*self.v_L_diag + (1-self.beta2)*L_diag**2
        m_hat_L_diag = self.m_L_diag/(1-np.power(self.beta1,self.t+1))
        v_hat_L_diag = self.v_L_diag/(1-np.power(self.beta2,self.t+1))
        update_L_diag = self.step*m_hat_L_diag / (np.sqrt(v_hat_L_diag)+epsilon)

        self.m_A = self.beta1*self.m_A + (1-self.beta1)*A
        self.v_A = self.beta2*self.v_A + (1-self.beta2)*A**2
        m_hat_A = self.m_A/(1-np.power(self.beta1,self.t+1))
        v_hat_A = self.v_A/(1-np.power(self.beta2,self.t+1))
        update_A = self.step*m_hat_A / (np.sqrt(v_hat_A)+epsilon)

        return [update_mu, update_L, update_L_diag, update_A]


def Q_est(mu, L, L_diag, log_A, lam, S1, S2, En, nu, n_obs, beta, mu_orig, sigma_orig, d, mus_orig, com):

    N = log_A.shape[0]
    d = mu.shape[1]
    t = 1+jnp.sum(En)

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

        # summed -= 0.0001*jnp.linalg.norm(mu - com + epsilon)**2
        
    return -jnp.sum(summed)/t 

def get_L(x, y):
    return jnp.tril(jnp.diag(jnp.exp(x) + epsilon) + jnp.tril(y,-1))

def get_L_inv(L):
    return jnp.linalg.inv(L)

def get_sig_inv(L):
    return L @ L.T

def get_sub_l(L):
    return L.flatten()/jnp.linalg.norm(L.flatten())

def get_mus(mu):
    return jnp.outer(mu,mu)

def get_ld(L):
    return -2 * jnp.sum(L)

def Q_j(S1, lam, sig_inv, mu, mu_orig, sigma_orig, S2, n_obs, mus, mus_orig, nu, ld, En, log_A, beta):
    summed = 0
    summed += (S1 + lam * mu_orig).dot(sig_inv).dot(mu) 
    summed += (-1/2) * jnp.trace( (sigma_orig + S2 + lam * mus_orig + (lam + n_obs) * mus) @ sig_inv ) 
    summed += (-1/2) * (nu + n_obs + 3 + 2) * ld
    summed += jnp.sum((En + beta - 1) * nn.log_softmax(log_A)) 
    return summed

## http://gregorygundersen.com/blog/2020/12/12/group-multivariate-normal-pdf/
def multiple_logpdfs(x, means, covs):
    # Thankfully, NumPy broadcasts `eigh`.
    vals, vecs = np.linalg.eigh(covs)

    # Compute the log determinants across the second axis.
    logdets    = np.sum(np.log(vals), axis=1)

    # Invert the eigenvalues.
    valsinvs   = 1./vals
    
    # Add a dimension to `valsinvs` so that NumPy broadcasts appropriately.
    Us         = vecs * np.sqrt(valsinvs)[:, None]
    devs       = x - means

    # Use `einsum` for matrix-vector multiplications across the first dimension.
    devUs      = np.einsum('ni,nij->nj', devs, Us)

    # Compute the Mahalanobis distance by squaring each term and summing.
    mahas      = np.sum(np.square(devUs), axis=1)
    
    # Compute and broadcast scalar normalizers.
    dim        = len(vals[0])
    log2pi     = np.log(2 * np.pi)
    return -0.5 * (dim * log2pi + mahas + logdets)