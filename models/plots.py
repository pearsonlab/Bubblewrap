import os
import sys

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

import numpy as np
import pandas as pd


def ewma(data, com):
    return np.array(pd.DataFrame(data=dict(data=data)).ewm(com).mean()['data'])


def saveplot():
    zp2016 = np.load(sys.argv[1])
    vjf = np.load(sys.argv[2])
    bwrap = np.load(sys.argv[3])
    # breakpoint()

    plt.figure(figsize=(8,4))
    nn = zp2016.shape[0]
    x = np.arange(nn-bwrap.shape[0], nn)

    plt.plot(x, bwrap, color='#FF4400', alpha=0.2)
    plt.plot(zp2016, color='purple', alpha=0.2)
    plt.plot(vjf, color='grey', alpha=0.2)

    ax = plt.gca()
    ax.plot(x, ewma(bwrap, 100), color='#FF4400', label='Bubblewrap')
    ax.plot(ewma(vjf, 100), color='black', label='VJF')
    ax.plot(ewma(zp2016, 100), color='purple', label='ZP (2016)')
    ax.ticklabel_format(useOffset=False, style='plain')

    ax.set_xlabel('time step')
    ax.set_ylabel('log probability')
    ax.set_title('predictive log probability at each time step')
    ax.legend()

    ax.set_ylim([float(sys.argv[4]), float(sys.argv[5])])

    # plt.savefig('Fig2c_1.svg', bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    if len(sys.argv) == 6:
        saveplot()
    else:
        print(f"usage: python {sys.argv[0]} (ZP2016-filename) (VJF-filename) (bubblewarp-filename) (ylim-min) (ylim max)", file=sys.stderr)