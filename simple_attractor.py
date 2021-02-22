import numpy as np
import matplotlib.pylab as plt
from scipy.integrate import solve_ivp

## Derivatives/datagen for simple 3D attractor
def lorenz(t, xyz, var):
    # returns value of derivatives
    x, y, z = xyz
    return var['sigma'] * (y - x), x * (var['rho'] - z) - y, x * y - var['beta'] * z


if __name__ == "__main__":

    ## Lorenz parameters
    rho = 28
    sigma = 10
    beta = 8/3
    var = {'rho': rho, 'sigma': sigma, 'beta': beta}
    
    ## initial conditions
    x0 = 1
    y0 = 1
    z0 = 1

    ## desired time points
    t = 100
    tn = 1000

    ## use scipy ode integrator
    soln = solve_ivp(lorenz, (0, t), (x0, y0, z0), args=(var,), dense_output=True)
    t = np.linspace(0, t, tn)
    x, y, z = soln.sol(t)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    cmap = plt.cm.plasma
    for i in range(0,tn):
        ax.scatter(x[i], y[i], z[i], color=cmap(i/tn))

    plt.show()