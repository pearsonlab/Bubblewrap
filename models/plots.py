import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ewma(data, com):
    return np.array(pd.DataFrame(data=dict(data=data)).ewm(com).mean()['data'])


def saveplot():
    zp2016 = np.load(sys.argv[1])
    vjf = np.load(sys.argv[2])
    bwrap = np.load(sys.argv[3])

    plt.ioff()
    nn = vjf.shape[0]

    plt.plot(np.linspace(100, nn - 1, nn - 100), zp2016, color='red', alpha=0.2)
    plt.plot(vjf, color='grey', alpha=0.2)
    plt.plot(bwrap, color='lightblue', alpha=0.2)

    ax = plt.gca()
    ax.plot(np.linspace(100, nn - 1, nn - 100), ewma(zp2016, 100), color='red', label='ZP2016')
    ax.plot(ewma(vjf, 100), color='black', label='vjf')
    ax.plot(ewma(bwrap, 100), color='blue', label='bubblewrap')
    ax.ticklabel_format(useOffset=False, style='plain')

    ax.set_xlabel('time step')
    ax.set_ylabel('log probability')
    ax.set_title('predictive log probability at each time step')
    ax.legend()

    ax.set_ylim([float(sys.argv[4]), float(sys.argv[5])])

    plt.savefig(f"{os.path.splitext(os.path.basename(sys.argv[3]))[0]}.svg")

    plt.show()

if __name__ == '__main__':
    if len(sys.argv) == 6:
        saveplot()
    else:
        print(f"usage: python {sys.argv[0]} (ZP2016-filename) (VJF-filename) (bubblewarp-filename) (ylim-min) (ylim max)", file=sys.stderr)