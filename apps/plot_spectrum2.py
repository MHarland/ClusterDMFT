#!/usr/bin/env pytriqs
from ClusterDMFT.cdmft import CDmft
import sys
from matplotlib import pyplot as plt, cm
from numpy import array

for n, arch in enumerate(sys.argv[1:]):
    c = CDmft(archive = arch)
    s = c.load('spectrum')
    g_e = list()
    i = 0
    eps = float(10**(-12))
    while i < len(s):
        if not(g_e) or abs(s[i] - s[i-1]) > eps:
            g_e.append([s[i], 1])
        else:
            g_e[-1][1] += 1
        i += 1
    g_e = array(g_e)
    fig, ax = plt.subplots()
    for pt in g_e:
        ax.plot([pt[0]]*2, [0, pt[1]], color = 'blue')
    ax.set_xlabel('$\omega$')
    ax.set_ylabel('degeneracy')
    ax.set_xlim(min(g_e[:, 0]) - .05*max(g_e[:, 0]), max(g_e[:, 0]) * 1.05)
    ax.set_ylim(0, max(g_e[:, 1]) * 1.05)
    plt.tight_layout()
    filename = arch[:-3]+'_spec.pdf'
    plt.savefig(filename, dpi = 300)
    plt.close()
    print filename+' ready'
