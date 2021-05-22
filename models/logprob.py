import os
import sys

import numpy as np
import scipy.io as sio
import torch
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from tqdm import trange
from vjf import online


def import_lorenz_vdp(filename):
    data = np.load(filename)

    xs = data['x']  # state
    ys = data['y']  # observation
    us = data['u']  # control input
    xdim = xs.shape[-1]
    ydim = ys.shape[-1]
    udim = us.shape[-1]
    return ys, us, xdim, ydim, udim

def import_jpca(filename):
    data = np.load(filename)

    xs = None  # state
    ys = data  # observation
    ys = ys.reshape(1, -1, 6)
    print(ys.shape)
    us = np.zeros((ys.shape[0], ys.shape[1], 1))  # control input
    xdim = 6
    ydim = ys.shape[-1]
    udim = us.shape[-1]
    return ys, us, xdim, ydim, udim


def import_neuropixel(filename):

    matdict = sio.loadmat('WaksmanwithFaces_KS2.mat', squeeze_me=True)
    spks = matdict['stall']
    spks = spks[..., None]
    xs = None  # state
    ys = spks  # observation
    us = np.zeros((ys.shape[0], ys.shape[1], 1))  # control input
    xdim = 3  # hidden state dimension
    ydim = ys.shape[-1]
    udim = us.shape[-1]
    return ys, us, xdim, ydim, udim


## fit and save log probability
def train(ys, us, xdim, ydim, udim):
    device = 'cpu'
    likelihood = 'gaussian'  # Gaussian observation
    dynamics = 'rbf'  # RBF network dynamic model
    recognizer = "mlp"  # MLP recognitiom model
    rdim = 100  # number of RBFs
    hdim = 100  # number of MLP hidden units

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

    yy = torch.from_numpy(ys).float().to(device)
    uu = torch.from_numpy(us).float().to(device)

    EXAMPLE = 0  # the example for calculating the likelihood

    # yy = torch.from_numpy(ys[EXAMPLE]).float()
    # uu = torch.from_numpy(us[EXAMPLE]).float()

    q = None

    logprobs = []
    distances = []

    for trial in range(yy.shape[0]):
        batch_size = 1
        filtering_mu = torch.zeros(batch_size, xdim, device=device)
        filtering_logvar = torch.zeros(batch_size, xdim, device=device)
        q = filtering_mu, filtering_logvar
        logprob_per_trial = []

        for time in trange(yy.shape[1]):

            y = yy[trial, time].unsqueeze(0)
            u = uu[trial, time].unsqueeze(0)

            if trial % 10 == 0:
                filtering_mu, filtering_logvar = q

                mu_f = filtering_mu[0].detach().cpu().numpy().T
                var_f = filtering_logvar[EXAMPLE].detach().exp().cpu().numpy().T
                Sigma_f = np.eye(xdim) * var_f

                N = 100
                x = multivariate_normal(mu_f.flatten(), Sigma_f).rvs(size=N).astype(np.float32)
                x = torch.from_numpy(x).to(device)
                x += mdl.system.velocity(x) + mdl.system.noise.var ** 0.5 * torch.randn_like(x)
                y_tilde = mdl.decoder(x).detach().cpu().numpy()
                y_var = mdl.likelihood.logvar.detach().exp().cpu().numpy().T
                logprob = logsumexp([multivariate_normal(y_t, y_var).logpdf(y.cpu().numpy()) for y_t in y_tilde]) - np.log(
                    N)

                logprob_per_trial.append(logprob)
                distances.append(np.linalg.norm(y_tilde - y[EXAMPLE].cpu().numpy(), axis=-1).mean())

            q, loss = mdl.feed((y, u), q)

        if len(logprob_per_trial) > 0:
            logprobs.append(logprob_per_trial)

    np.save(f"logprob_{os.path.splitext(os.path.basename(sys.argv[2]))[0]}.npy", np.array(logprobs[0]))

if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'lorenz':
        train(*import_lorenz_vdp(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == 'vdp':
        train(*import_lorenz_vdp(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == 'jPCA':
        train(*import_jpca(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == 'neuropixel':
        train(*import_neuropixel(sys.argv[2]))
    else:
        print(f"usage: python {sys.argv[0]} (lorenz|vdp|jPCA|neuropixel filename)", file=sys.stderr)