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
        # self.sigma = np.zeros((self.N,self.d,self.d), dtype="float32")
        # for n in np.arange(self.N):
        #     self.sigma[n] = np.diagflat(np.ones((self.d), dtype="float32"))        #assuming isotropic
        # index/mask array for children per node
        self.children = -1 * np.ones((self.N, self.max_child), dtype=int)
        # or
        self.edges = np.diagflat(np.ones(self.N))
        ## TODO: make an actual graph structure using networkx?

        # observations of the data; M is how many to keep in history
        self.obs = Observations(self.d, M=M)

        # HMM variables
        # self.A = (1/self.N) * np.ones((self.N,self.N)) #self.edges.copy()      #FIXME
        self.A = np.ones((self.N,self.N)) - np.eye(self.N)#*0.99999
        self.A /= np.sum(self.A, axis=1)
        
        self.B = (1/self.N) * np.ones((self.N))
        # self.Q = 0
        # self.gamma = None
        self.alpha = np.zeros((self.N))
        self.En = (1/self.N) * np.ones((self.N,self.N))

        # self.S1 = 0
        # self.S2 = 0
        
    def init_nodes(self):
        # set initial centers of nodes distributed across space of observations
        sl = [slice(0, floor(np.sqrt(self.N)))] * self.d       #FIXME
        self.mu = np.mgrid[sl].reshape((self.d, self.N)).astype("float32").T

        com = center_mass(self.mu)
        obs_com = center_mass(self.obs.saved_obs)

        self.mu -= com
        scale = np.max(np.abs(self.obs.saved_obs - obs_com))*2  / self.d / floor(np.sqrt(self.N))
        scale *= 15
        self.mu *= scale
        self.mu += obs_com

        # breakpoint()

        # self.sigma *= scale/4
        self.fullSigma = np.zeros((self.N,self.d,self.d), dtype="float32")
        for n in np.arange(self.N):
            self.fullSigma[n] = np.diagflat(self.sigma_scale*(1/scale)*np.ones((self.d), dtype="float32"))   

        # copies for later
        self.mu_orig = self.mu.copy()
        self.sigma_orig = self.sigma.copy()
        self.fullSigma_orig = self.fullSigma.copy()
        self.A_orig = self.A.copy()

        # B = np.zeros((self.N))
        # for n in np.arange(self.N):
        #     B[n] = mvn.pdf(self.obs.curr, mean=self.mu[n], cov=self.fullSigma[n])
        # B[B<epsilon] = epsilon

        prior = (1/self.N)*np.ones(self.N)
        self.alpha = (self.lam_0) * prior #B
        # print('----------- ', self.alpha)

        # self.alpha = np.ones(self.N) * (1/self.N)
        self.last_alpha = self.alpha
        self.alpha_sum = self.alpha

        self.t = 1


        self.lam = (self.lam_0) * prior #self.alpha
        self.nu = prior * (self.nu_0) # + self.d + 1)

        self.n_obs = 0 #self.lam # ??

        self.fullSigma_orig *= prior[:,None,None]
        self.mus_orig = np.zeros((self.N,self.d,self.d))
        for n in np.arange(self.N):
            self.mus_orig[n] = np.outer(self.mu[n], self.mu[n])

        # ###
        # self.S1 = self.alpha[:,None] * self.mu #self.alpha[:,np.newaxis] * self.obs.curr # np.ones(self.N)[:,np.newaxis] * self.obs.curr
        # mus = np.zeros_like(self.fullSigma)
        # for n in np.arange(self.N):
        #     mus[n] = np.outer(self.mu[n], self.mu[n])
        # self.S2 = prior[:,None,None] * self.fullSigma + self.alpha[:,None,None] * mus #self.alpha[:,np.newaxis,np.newaxis] * (self.obs.curr[:,np.newaxis] * self.obs.curr.T) # np.ones(self.N)[:,np.newaxis,np.newaxis] * (self.obs.curr[:,np.newaxis] * self.obs.curr.T)

        self.S1 = np.zeros((self.N,self.d))
        self.S2 = np.zeros((self.N,self.d,self.d))

        # print(self.S1[0])
        # print(self.S2[0])
        # print('---------------')



        # self.update_S()
        self.update_ss()


        # breakpoint()

        self.Q_list = []

    def observe(self, x):
        # Get new data point and update observation history
        self.obs.new_obs(x)

        ## WHERE?
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

        # breakpoint()

    def grad_jax(self):
        # get gradients of all variables at once
        # then take a step 

        args = [self.obs.curr, self.A, self.mu, self.sigma, self.lam]
        A_grad = self.grad_A(*args) * self.step * 1e-2
        mu_grad = self.grad_mu(*args) * self.step *1e2
        sigma_grad = self.grad_sigma(*args) * self.step 

        # breakpoint()

        self.A -= A_grad
        self.mu -= mu_grad
        self.sigma -= sigma_grad

        # print('######################')
        # l = total_loss(*args)
        # print(l)
        # print(np.sum(A_grad), np.sum(mu_grad), np.sum(sigma_grad))
        # print(np.sum(self.A), np.sum(self.mu), np.sum(self.sigma))

        # if np.isnan(l).any():
        #     print('AAAHHH NANS')
        #     breakpoint()

        self.step /= 1.001
        self.t += 1

    def grad_gm(self):
        breakpoint()
        # params_init = log_component_weights_init, component_mus_init, log_component_scales_init
        # grad = self.dloss_mixture_weights(params_init, data_mixture)

        # self.loss = total_loss(self.obs.curr, self.A, self.mu, self.sigma)

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

    # def prune(self):
    #     # Prune graph; soft-thresholding

    def update_gamma(self):
        # Compute new update matrix gamma
        # gamma_{l,h}(T) = B_{h,x_T} A_{l,h} / sum_{i,j}(B_{j,x_T} A_{i,j} alpha_i(T-1))

        self.gamma = self.B * self.A / (self.alpha.dot(self.A).dot(self.B)) # + epsilon)
        # breakpoint()
        # print('---------------', np.sum(self.A-self.A_orig))

        # if np.isnan(self.gamma).any():
        #     breakpoint()
    
    def update_alpha(self):
        # Compute new posterior marginal probabiltiies alpha
        # alpha_j(T) = sum_i (gamma_{i,j}(T) alpha_i(T-1))

        self.alpha = self.last_alpha.dot(self.gamma)
        self.alpha[self.alpha<epsilon] = epsilon
        self.alpha /= np.sum(self.alpha)
        # print('             ', self.alpha)

        # if np.isnan(self.alpha).any():
        #     breakpoint()

        # self.alpha_sum += self.alpha

    def update_En(self):
        # Compute new N matrix [N_{i,j} = sum_t p(z_t=j, z_{t+1} = 1 | x_{1:T})]
        # N_{i,j} = (1/T) gamma_{i,j}(T) alpha_i(T) + (1-1/T) N_{i,j}(T-1)

        self.En = (1/self.t) * self.gamma * self.alpha[:,np.newaxis] + (1-1/self.t) * self.En
        # print(self.En)

        # if np.isnan(self.En).any():
        #     breakpoint()

    def update_A(self):
        # Compute updated state evolution matrix A using N
        # A_{i,j} = N_{i,j} / sum_j N_{i,j}

        self.A = self.En / (np.sum(self.En, axis=1)[:,np.newaxis] + epsilon)

        # if np.isnan(self.A).any():
        #     breakpoint()

    def update_S(self):
        # update S tensor
        # S1(T+1) = (1-eps)*S1(T) + eps*alpha(T) @ x
        # S2 sim but @ x xT
        # S1 shape: (N,d)
        # S2 shape: (N,d,d)
        self.eps = 1/(self.t+1)

        # if self.t<=1:
        #     self.S1 = self.S1 + self.alpha[:,np.newaxis] * self.obs.curr
        #     self.S2 = self.S2 + self.alpha[:,np.newaxis,np.newaxis] * (self.obs.curr[:,np.newaxis] * self.obs.curr.T)
        # else:
        # self.S1 = (1 - self.eps)*self.S1 + self.eps*self.alpha[:,np.newaxis] * self.obs.curr
        # self.S2 = (1 - self.eps)*self.S2 + self.eps*self.alpha[:,np.newaxis,np.newaxis] * (self.obs.curr[:,np.newaxis] * self.obs.curr.T)

        # self.n_obs = (1 - self.eps)*self.n_obs + self.eps*self.alpha

        self.S1 += self.alpha[:,np.newaxis] * self.obs.curr
        self.S2 += self.alpha[:,np.newaxis,np.newaxis] * (self.obs.curr[:,np.newaxis] * self.obs.curr.T)
        self.n_obs += self.alpha
        
        print(np.max(self.n_obs))

        # print(self.S1[0])
        # print(self.S2[0])
        # print('---------------')

    def update_ss(self):
        # Update sufficient statistics
            # self.mu = self.S1 / self.alpha_sum[:,None]
            # mus = np.zeros_like(self.S2)
            # for n in np.arange(self.N):
            #     mus[n] = np.outer(self.mu[n], self.mu[n])
            # self.fullSigma = self.S2 / self.alpha_sum[:,None,None] - mus
            # self.sigma = self.fullSigma[:,0,0]
        # self.sigma[self.sigma<0] = epsilon

        # breakpoint()

        # self.lam = (1 - self.eps)*self.lam + self.eps*self.alpha
        # self.nu = (1 - self.eps)*self.nu + self.eps*self.alpha

        ###
        # self.mu = self.S1 / self.lam[:,None]
        # # breakpoint()
        
        # self.fullSigma = (self.S2 - self.lam[:,None,None]*mus) / (self.nu[:,None,None])
        # breakpoint()

        self.mu = (self.lam[:,None] * self.mu_orig + self.S1) / (self.lam + self.n_obs)[:,None]
        mus = np.zeros_like(self.S2)
        for n in np.arange(self.N):
            mus[n] = np.outer(self.mu[n], self.mu[n])
        self.fullSigma = (self.fullSigma_orig + self.lam[:,None,None]*self.mus_orig + self.S2 - (self.lam + self.n_obs)[:,None,None] * mus) / (self.nu + self.d + 1 + self.n_obs)[:,None,None]

        # breakpoint()

    def update_Q(self):
        # Compute new estimate for expected log-likelihood Q
        # Q_{T+1} = Q_{T} + eps_{T+1} [sum_j alpha_j(T) log P(x_T | z_T=j) - Q_{T}]

        # self.Q = self.Q + self.eps * (np.sum(self.last_alpha @ self.B) - self.Q)
        ## return self.Q?

        self.Q = -Q_est(self.obs.curr, self.mu, self.sigma, self.Q, self.alpha, self.eps)
        self.Q_list.append(self.Q)
        # print(self.Q)
        if np.isnan(self.Q).any():
            breakpoint()


    def grad_theta(self):
        # Take step in theta parameters using gradient Q update
        # theta -= eta * grad_{theta} Q_{T}

        args = [self.obs.curr, self.mu, self.sigma, self.Q, self.alpha, self.eps]
        mu_grad = self.grad_mu(*args) * self.step * 1e3
        sigma_grad = self.grad_sigma(*args) * self.step 

        if np.isnan(mu_grad).any() or np.isnan(sigma_grad).any():
            print('-------------- nan in grad')
            breakpoint()

        self.mu -= mu_grad
        self.sigma -= sigma_grad

        self.step /= 1.001
        self.t += 1
        self.eps /= 1.001       #FIXME


def Q_est(x, mu, sigma, Q, alpha, eps):
    summed = 0
    for n in np.arange(mu.shape[0]):
        summed += alpha[n]*jmvn.logpdf(x, mean=mu[n], cov=sigma[n])

    return -(Q + eps*(summed - Q))

# def gaussian_dist(x, mu, sigma):
#     # multivariate, single node
#     coeff = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(sigma)**(1/2)) )
#     expon = -0.5*((x - mu).T.dot(np.linalg.inv(sigma)).dot(x - mu))
#     return coeff * np.exp(expon)

# def loss(x, A, mu, sigma):
#     for n in np.arange(mu.shape[0]):
#         w = A[n] @ gaussian_dist(x, mu[n], sigma[n])
#         print(w.shape)

def total_loss(x, A, mu, sigma, lam):
    summed = 0
    nodes = A.shape[0]
    children = A.shape[1] #tho A is currently square
    dim = mu.shape[1]
    for n in jnp.arange(nodes):
        for c in jnp.arange(children):
            summed += A[n,c]*mvn.pdf(x,mean=mu[c],cov=sigma[c])
            # print(mu[c], sigma[c], A[n,c])
            # print(summed)
            # summed += A[n,c]*jnp.sum(gaussian_dist(x, mu[c], sigma[c]))
    # summed += lam*jnp.sum(A)
    summed += eps
    return -jnp.log(summed) - lam*jnp.sum(A)

