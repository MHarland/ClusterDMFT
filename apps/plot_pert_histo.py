#!/usr/bin/env pytriqs
from ClusterDMFT.cdmft import CDmft
import sys
from matplotlib import pyplot as plt, cm
from numpy import loadtxt

max_plot_order = 0
for n, filename in enumerate(sys.argv[1:]):
    data = loadtxt(filename)
    max_order = 0
    cum = 0
    while cum < 1.0 and max_order < 100:
        cum = data[max_order, 2]
        max_order += 1
    if max_order > max_plot_order: max_plot_order = max_order
    plt.plot(data[:, 0], data[:, 1], label = filename[17:-4], color = cm.jet(n /float(len(sys.argv[1:]) - 1)))

plt.gca().set_xlim(0, max_plot_order)
plt.legend()
plt.savefig('pert_histo.pdf', dpi = 300)
