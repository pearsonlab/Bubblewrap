#!/usr/bin/env python
# coding: utf-8

# ## 1. Install VJF (for more information, visit https://github.com/catniplab/vjf)

# # 


# get_ipython().system('rm -rf /tmp/vjf/')
# get_ipython().system('git clone https://github.com/catniplab/vjf.git /tmp/vjf/')


# # 


# get_ipython().system('git --git-dir=/tmp/vjf/.git --work-tree /tmp/vjf/ checkout 0eec61e91c29cf9a44b48c2a6694234b4404a2b3')


# :


import sys
sys.path.append("/tmp/vjf")


# ## 2. Import the packages / functions

# :


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from tqdm.notebook import trange
import copy
import torch
import vjf
from vjf import online

# get_ipython().run_line_magic('matplotlib', 'inline')


# ## 3. Import the dataset
# ### (run ONLY one of the 6 cells below, depending on the dataset you'd like to run with)
# - Van der Pol dataset
# - Lorenz attractor dataset
# - Monkey reach (jPCA) dataset
# - Wide-field calcium dataset
# - Mouse video dataset
# - Neuropixels dataset

# #### run the cell below to load the simulated Van der Pol dataset 

# 


# The data loading below is the vdp with (1 trajectory, 2dim, 500 to 20500 time points, with noise sd=0.05)
# Use any of the simulated vdp dataset. 
# You can generate the simulated data using datagen.py (more info in Readme)
# data = np.load('vdp_1trajectories_2dim_500to20500_noise0.05.npz')

# xs = data['x']  # state
# ys = data['y']  # observation
# us = data['u']  # control input
# xdim = xs.shape[-1]
# ydim = ys.shape[-1]
# udim = us.shape[-1]


# #### run the cell below to load the simulated Lorenz attractor dataset 

# :


# The data loading below is the lorenz attractor with (1 trajectory, 3dim, 500 to 20500 time points, with noise sd=0.05)
# Use any of the simulated lorenz dataset. 
# You can generate the simulated data using datagen.py (See README.md for more info)
# data = np.load('lorenz_1trajectories_3dim_500to20500_noise0.2.npz')

# xs = data['x']  # state
# ys = data['y']  # observation
# us = data['u']  # control input
# xdim = xs.shape[-1]
# ydim = ys.shape[-1]
# udim = us.shape[-1]


# #### run the cell below to load the reduced Monkey reach (jPCA) dataset

# :


# load the reduced Monkey reach dataset (See README.md for how the reduced dataset was generated.)
# data = np.load('jpca_reduced.npy')

# xs = None  # state
# ys = data.T  # observation
# ys = ys[None, ...]
# us = np.zeros((ys.shape[0], ys.shape[1], 1))  # control input
# xdim = 6
# ydim = ys.shape[-1]
# udim = us.shape[-1]

# breakpoint()
# #### run the cell below to load the reduced Wide-field calcium imaging dataset

# :


# load the reduced Wide-field calcium dataset (See README.md for how the reduced dataset was generated.)
# data = np.load('widefield_reduced.npy')

# xs = None
# ys = data.T[None, ...]
# us = np.zeros((ys.shape[0], ys.shape[1], 1))
# xdim = ys.shape[-1]
# ydim = ys.shape[-1]
# udim = us.shape[-1]


# #### run the cell below for the reduced Mouse video dataset

# :


# load the reduced Mouse video dataset (See README.md for how the reduced dataset was generated.)
# data = np.load('reduced_mouse.npy')

# xs = None
# ys = data.T[None, ...]
# us = np.zeros((ys.shape[0], ys.shape[1], 1))
# xdim = ys.shape[-1]
# ydim = ys.shape[-1]
# udim = us.shape[-1]


# #### run the cell below for the example Neuropixels dataset

# :


# load the reduced Neuropixels dataset with the desired latent dimensions
# (See README.md for how the reduced dataset was generated.)
data = np.load('neuropixel_reduced.npz')['ssSVD10'] # you can change this to 'ssSVD20'

xs = None
ys = data.T[None, ...]
us = np.zeros((ys.shape[0], ys.shape[1], 1))
xdim = ys.shape[-1]
ydim = ys.shape[-1]
udim = us.shape[-1]


# ## 4. Check your data dimension
# It should be
# `number of trials x number of time points x number of latent dimensions`

# :


print(ys.shape)


# ## 5. Random Seed 

# :


import random
import torch

def set_seed(seed=None, seed_torch=True):
    if seed is None:
        seed = np.random.choice(2 ** 32)
        random.seed(seed)
        np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    print(f'Random seed {seed} has been set.')

SEED = 2021
set_seed(seed=SEED)


# ## 6. Training the model and calculating the log probs

# :


device = 'cpu'


# :


likelihood = 'gaussian'  # Gaussian observation
dynamics = 'rbf'  # RBF network dynamic model
recognizer = "mlp"  # MLP recognitiom model
rdim = 50  # number of RBFs
hdim = 100  # number of MLP hidden units


# :


def diagonal_normal_logpdf(mean, variance, sample):
    mean = mean.flatten()
    variance = variance.flatten()
    sample = sample.flatten()
    
    assert len(mean) == len(variance) == len(sample), f"inconsistent shape: {mean.shape}, {variance.shape}, {sample.shape}"
    
    logprobs = []
    for i in range(len(sample)):
        x = sample[i]
        m = mean[i]
        v = variance[i]
        logprobs.append(-0.5 * ((x - m) ** 2 / v + np.log(2 * np.pi * v)))
    return sum(logprobs)


# :


yy = torch.from_numpy(ys).float().to(device)
uu = torch.from_numpy(us).float().to(device)

q = None


# :


S = 1000  # number of trajectories
T = 10   # length of each trajectory
P = 1    # calculate T-step-ahead predictive distribution every P steps


# :


logprobs = []
logprob_trajectories = []
distances = []
trial = 0


for trial in trange(yy.shape[0]):
    batch_size = 1
    filtering_mu = torch.zeros(batch_size, xdim, device=device)
    filtering_logvar = torch.zeros(batch_size, xdim, device=device)
    q = filtering_mu, filtering_logvar
    logprob_per_trial = []
    logprob_trajectories_per_trial = []
    
    mdl = online.VJF(
        config=dict(
            resume=False,
            xdim=xdim,
            ydim=ydim,
            udim=udim,
            Ydim=udim,
            Udim=udim,
            rdim=rdim,
            hdim=hdim,
            lr=1e-3,
            clip_gradients=5.0,
            debug=True,
            likelihood=likelihood,  # 
            system=dynamics,
            recognizer=recognizer,
            C=(None, True),  # loading matrix: (initial, estimate)
            b=(None, True),  # bias: (initial, estimate)
            A=(None, False),  # transition matrix if LDS
            B=(np.zeros((xdim, udim)), False),  # interaction matrix
            Q=(1.0, True),  # state noise
            R=(1.0, True),  # observation noise
        )
    ).to(device)
    
    for time in trange(yy.shape[1]):

        y = yy[trial, time].unsqueeze(0)
        u = uu[trial, time].unsqueeze(0)
        
        filtering_mu, filtering_logvar = q

        mu_f = filtering_mu[0].detach().cpu().numpy().T
        var_f = filtering_logvar[0].detach().exp().cpu().numpy().T
        Sigma_f = np.eye(xdim) * var_f

        x = multivariate_normal(mu_f.flatten(), Sigma_f).rvs(size=S).astype(np.float32)
        x = torch.from_numpy(x).to(device)
        x += mdl.system.velocity(x) + mdl.system.noise.var ** 0.5 * torch.randn_like(x)
        y_tilde = mdl.decoder(x).detach().cpu().numpy()

        y_var = mdl.likelihood.logvar.detach().exp().cpu().numpy().T
        sample_logprobs = [diagonal_normal_logpdf(y_, y_var, y.cpu().numpy()) for y_ in y_tilde]
        logprob = logsumexp(sample_logprobs) - np.log(S)

        logprob_per_trial.append(logprob)
        distances.append(np.linalg.norm(y_tilde - y[0].cpu().numpy(), axis=-1).mean())
        
        
        if time % P == 0 and time + T < yy.shape[1]:
            # rolling-predict T-1 more steps
            with torch.no_grad():
                trajectory_logprobs = [logprob]
                mdl_copy = copy.deepcopy(mdl)  # just to make sure we're not touching the original model

                for tprime in range(time + 1, time + T):
                    y_tprime = yy[trial, tprime].unsqueeze(0)
                    u_tprime = uu[trial, tprime].unsqueeze(0)

                    x += mdl_copy.system.velocity(x) + mdl_copy.system.noise.var ** 0.5 * torch.randn_like(x)
                    y_tilde = mdl_copy.decoder(x).detach().cpu().numpy()
                    # y_var didn't change

                    # cumulative sum on sample logprobs
                    sample_logprobs = [diagonal_normal_logpdf(y_, y_var, y_tprime.cpu().numpy()) for y_ in y_tilde]
                    # sample_logprobs = [a + b for a, b in zip(sample_logprobs, new_sample_logprobs)]

                    logprob = logsumexp(sample_logprobs) - np.log(S)
                    trajectory_logprobs.append(logprob)

                logprob_trajectories_per_trial.append(trajectory_logprobs)

        q, loss = mdl.feed((y, u), q)

    logprobs.append(logprob_per_trial)
    logprob_trajectories.append(logprob_trajectories_per_trial)


# ## 7. Save the log probabilities 

# :


np.save('logprob_vjf_neuropixels_seed2021_P1.npy', np.array(logprob_trajectories[0]))


# :


trajectory_logprobs = np.array(logprob_trajectories[0]) #np.load('logprob_vjf_widefield_seed44.npy')


# ## 8. Plotting the log probability 

# :


import pandas as pd
def ewma(data, com):
    return np.array(pd.DataFrame(data=dict(data=data)).ewm(com).mean()['data'])


# :


plt.figure(figsize=(9, 6))

for i in range(trajectory_logprobs.shape[-1]):
    curve = ewma(trajectory_logprobs[:, i], 100)
    plt.plot(np.arange(len(curve)) * P, curve, label=f"{i+1} step{'s' if i > 0 else ''} ahead")
    
plt.legend(bbox_to_anchor=(1.01, 0.95))
plt.ylabel("log probability")
plt.xlabel("time")
# plt.ylim([-300, 0])


# ## 9. Compute the mean and std of the last half of the time points (For Table1)

# :


nn = trajectory_logprobs.shape[0]
trajectory_logprobs[nn//2:, 0].mean(), trajectory_logprobs[nn//2:, 0].std()


# ## 10. Compute the mean and std of the last half of the time points (for the new figure)
# ### (1 step prediction to 10 step predictions)

# :


nn = trajectory_logprobs.shape[0]
for i in range(10):
    print(f"{i+1} step:", trajectory_logprobs[nn//2:, i].mean(), trajectory_logprobs[nn//2:, i].std())


# :


prediction = np.zeros((6, 3))
i = 0
for t in range(500):
    if t in [0, 1, 2, 3, 4, 9]:
        prediction[i] = int(t+1), trajectory_logprobs[nn//2:, t].mean(), trajectory_logprobs[nn//2:, t].std()
        i += 1


# :


prediction

plt.show()


# :


np.save('tensteps_vjf_neuropixels_seed2021_P1.npy', prediction)


# 



