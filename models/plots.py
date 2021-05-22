import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ewma(data, com):
    return np.array(pd.DataFrame(data=dict(data=data)).ewm(com).mean()['data'])


def saveplot():
    data = np.load(sys.argv[1])
    plt.ioff()
    plt.plot(data, color='grey', alpha=0.2)

    ax = plt.gca()
    ax.plot(ewma(data, 100), color='black')
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.set_xlabel('time step')
    ax.set_ylabel('log probability')
    ax.set_title('predictive log probability at each time step')

    plt.savefig(f"{os.path.splitext(os.path.basename(sys.argv[1]))[0]}.pdf")
    # ax.set_ylim([-30,0])


if __name__ == '__main__':
    if len(sys.argv) == 2:
        saveplot()
    else:
        print(f"usage: python {sys.argv[0]} (filename)", file=sys.stderr)