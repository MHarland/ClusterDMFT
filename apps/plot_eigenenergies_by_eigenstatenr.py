#!/usr/bin/env pytriqs
from ClusterDMFT.cdmft import CDmft
import sys, numpy
from matplotlib import pyplot as plt, cm

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
archives = sys.argv[1:]
for n, arch in enumerate(sys.argv[1:]):
    c = CDmft(archive = arch)
    subspaces = c.load('eigensystems')
    energies = list()
    counter = 0
    for sub in subspaces:
        for energy in sub[0]:
            energies.append(energy)
            if numpy.allclose(0, energy):
                print counter
            counter += 1
    for i, energy in enumerate(energies):
        if i == 0:
            ax.plot([i]*2, [0,energy], label = arch[:-3], color = plt.cm.jet(n/float(max(len(archives)-1,1))))
        else:
            ax.plot([i]*2, [0,energy], color = plt.cm.jet(n/float(max(len(archives)-1,1))))
ax.set_xlim(-10,260)
ax.legend()
plt.savefig('energies_by_eigenstatenr.pdf', dpi = 300)
