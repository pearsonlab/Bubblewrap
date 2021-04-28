import numpy as np
import jax.numpy as jnp
from math import floor

from field.mesh import Observations
from field.utils import center_mass

from jax import jit, grad
import jax.scipy.stats
from jax.scipy.stats import multivariate_normal as jmvn
from scipy.stats import multivariate_normal as mvn
from jax.scipy.special import logsumexp as lse

from scipy.special import softmax

epsilon = 1e-10
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


## Working title: 'Graph quantized dynamical systems' (gqds)

class GQDS():
    def __init__(self, num, dim, seed=42, M=10, max_child=10, step=1e-6, lam=1, eps=1e-4, nu=1e-2, sigma_scale=1e6, beta=2, eta=1e-3):
        self.N = num            # Number of nodes
        self.d = dim            # dimension of the space
        self.seed = seed
        self.max_child = max_child
        self.step = step
        self.lam_0 = lam
        self.nu_0 = nu
        # self.eps = eps
        self.sigma_scale = sigma_scale
        self.beta = beta
        self.eta = eta

        # observations of the data; M is how many to keep in history
        self.obs = Observations(self.d, M=M)

        self.alpha = (1/self.N)*np.ones(self.N)

        
    def init_nodes(self):
        ### Compute initial ss based on observed data so far
        # set initial centers of nodes distributed across space of observations
        sl = [slice(0, floor(np.sqrt(self.N)))] * self.d       
        self.mu = np.mgrid[sl].reshape((self.d, self.N)).astype("float32").T

        com = center_mass(self.mu)
        obs_com = center_mass(self.obs.saved_obs)

        self.mu -= com
        scale = np.max(np.abs(self.obs.saved_obs - obs_com))*2  / self.d / floor(np.sqrt(self.N))
        scale *= 14
        self.mu *= scale
        self.mu += obs_com

        self.fullSigma = np.zeros((self.N,self.d,self.d), dtype="float32")
        for n in np.arange(self.N):
            self.fullSigma[n] = np.diagflat(self.sigma_scale*(1/scale)*np.ones((self.d), dtype="float32"))   

        ### Initialize model parameters (A,En,...)
        prior = (1/self.N)*np.ones(self.N)

        self.alpha = self.lam_0 * prior
        self.last_alpha = self.alpha.copy()
        self.lam = self.lam_0 * prior 
        self.nu = self.nu_0 * prior

        self.n_obs = 0

        self.A = np.ones((self.N,self.N)) - np.eye(self.N)#*0.99999
        self.A /= np.sum(self.A, axis=1)
        self.B = np.zeros((self.N))
        self.En = np.zeros((self.N,self.N)) #(1/self.N) * np.ones((self.N,self.N))

        self.S1 = np.zeros((self.N,self.d))
        self.S2 = np.zeros((self.N,self.d,self.d))

        self.fullSigma *= prior[:,None,None]
        self.mus_orig = np.zeros((self.N,self.d,self.d))
        for n in np.arange(self.N):
            self.mus_orig[n] = np.outer(self.mu[n], self.mu[n])

        self.L = np.linalg.cholesky(self.fullSigma)
        self.L_diag = np.zeros((self.N,self.d))
        for n in np.arange(self.N):
            self.L_diag[n] = np.log(np.diag(self.L[n]))
        #np.diagonal(self.L)
        # self.L = np.tril(L,-1)
        # breakpoint()
        self.L_lower = np.tril(self.L,-1) #np.zeros((self.N,self.d,self.d)) 
        # breakpoint()

        self.log_A = np.zeros((self.N,self.N))

        ### Save copies of mu, sigma, A for later comparison
        self.mu_orig = self.mu.copy()
        self.fullSigma_orig = self.fullSigma.copy()
        self.A_orig = self.A.copy()

        ## With these new values, update ss
        self.t = 1
        self.update_ss()

        ## Set up gradients
        self.grad_mu = jit(grad(Q_est, argnums=0))
        # self.grad_sigma = jit(grad(Q_est, argnums=1))
        self.grad_L_lower = jit(grad(Q_est, argnums=1))
        self.grad_L_diag = jit(grad(Q_est, argnums=2))
        self.grad_A = jit(grad(Q_est, argnums=3))

        self.Q_list = []
        self.A_diff = []

        ## for adam gradients
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


    def observe(self, x):
        # Get new data point and update observation history
        self.obs.new_obs(x)

        self.last_alpha = self.alpha.copy()

    def em_step(self):
        # take step in E and M; after observation

        # print(self.obs.curr)
        # print(self.mu)
        print(self.alpha)

        self.eps = 0.03 #1/(self.t+1)
        self.beta = 1 + 10/(self.t+1)

        self.update_B()
        self.update_gamma()
        self.update_alpha()
        self.update_En()
        # self.update_A()

        self.update_S()
        # self.update_ss()
        self.grad_Q()

        self.t += 1

    def update_B(self):
        # Compute posterior over graph
        for n in np.arange(self.N):
            self.fullSigma[n] = self.L[n] @ self.L[n].T
            try:
                self.B[n] = mvn.logpdf(self.obs.curr, mean=self.mu[n], cov=self.fullSigma[n] + epsilon+np.eye(self.d))
            except:
                print('problem in ', n, ' draw for B; passing...')
        
        max_Bind = np.argmax(self.B)
        self.B -= self.B[max_Bind]
        self.B = np.exp(self.B)

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
        

    def update_A(self):
        # Compute updated state evolution matrix A using N
        # A_{i,j} = N_{i,j} / sum_j N_{i,j}

        self.A = self.En / (np.sum(self.En, axis=1)[:,np.newaxis])


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

    
    def grad_Q(self):
        
        args = [self.mu, self.L_lower, self.L_diag, self.log_A, self.lam, self.S1, self.S2, self.eta, self.En, self.nu, self.n_obs, self.beta, self.mu_orig, self.fullSigma_orig, self.d, self.mus_orig]

        grad_mu = self.grad_mu(*args)
        grad_L = np.array(self.grad_L_lower(*args))
        grad_L_diag = np.array(self.grad_L_diag(*args))
        grad_A = self.grad_A(*args)

        
        if (grad_L_diag > 1e3).any():
            print('Large L_diag gradient')
            breakpoint()


        ## adam
        updates = self.run_adam(grad_mu, grad_L, grad_L_diag, grad_A)
        self.mu -= updates[0] 
        self.L_lower -= updates[1] 
        self.L_diag -= updates[2] 
        self.log_A -= updates[3] 

        self.A = softmax(self.log_A, axis=1)

        self.A_diff.append(self.A-self.A_orig)
        

        self.L = np.zeros((self.N,self.d,self.d)) #= np.diagonal(np.exp(self.L_diag)) + self.L_lower     #np.array(np.diagonal(self.L_diag))
        for n in np.arange(self.N):
            self.L[n] = np.diag(np.exp(self.L_diag[n])) + self.L_lower[n]

        
        Q = -Q_est(*args)
        print(self.t, Q)
        self.Q_list.append(Q)
        # print('-------------------------------after gradient')

        self.step /= 1.001

    def run_adam(self, mu,L,L_diag,A):
        ## inputs are gradients

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


from jax import nn
def Q_est(mu, L, L_diag, log_A, lam, S1, S2, eta, En, nu, n_obs, beta, mu_orig, sigma_orig, d, mus_orig):

    N = log_A.shape[0]
    d = mu.shape[1]
    t = jnp.sum(En)
    # sig_inv = jnp.linalg.inv(sigma)
    # chol_inv = jnp.linalg.inv(jnp.linalg.cholesky(sigma))
    # sig_inv = chol_inv.T @ chol_inv
    # ld = jnp.linalg.slogdet(sigma)[1]

    # el = jnp.tril((jnp.diagonal(jnp.exp(L_diag)+epsilon) + jnp.tril(L,-1)))

    # jnp.diag(L_diag)
    # el[1,1] = L

    summed = 0
    for j in jnp.arange(N):
        el = jnp.tril(jnp.diag(jnp.exp(L_diag[j]) + epsilon) + jnp.tril(L[j],-1))
        # L = jnp.linalg.cholesky(sigma[j])
        chol_inv = jnp.linalg.inv(el)
        sig_inv = chol_inv.T @ chol_inv #+ epsilon*jnp.diag(jnp.ones(d))
        # sig_inv = jnp.linalg.inv(sigma[j])
        ld = 2 * jnp.sum(L_diag[j])
        mus = jnp.outer(mu[j], mu[j])

        summed += (S1[j] + lam[j] * mu_orig[j]).dot(sig_inv).dot(mu[j])
        summed += (-1/2) * jnp.trace( (sigma_orig[j] + S2[j] + lam[j] * mus_orig[j] + (lam[j] + n_obs[j]) * mus) @ sig_inv )
        summed += (-1/2) * (nu[j] + n_obs[j] + d + 2) * ld
        summed += jnp.sum((En[j] + beta - 1) * nn.log_softmax(log_A[j])) #(A[:,j] - lse(A[:,j])))

    # last term sum over all elements
    # summed_last = np.sum( (En + beta - jnp.ones((N,N))) * (jnp.log(A+epsilon)) )

    return -summed/t #- summed_last #+ eta*jnp.sum((jnp.sum(A,axis=1)-jnp.ones(N)))