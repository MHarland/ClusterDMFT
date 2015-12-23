#!/usr/bin/env pytriqs
from ClusterDMFT.cdmft import CDmft
import sys, numpy
from matplotlib import pyplot as plt, cm

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
archives = sys.argv[1:]
for n, arch in enumerate(sys.argv[1:]):
    c = CDmft(archive = arch)
    probabilities = c.load('state_trace_contribs')
    for i, probability in enumerate(probabilities):
        if i == 0:
            ax.plot([i]*2, [0,probability], label = arch[:-3], color = plt.cm.jet(n/float(max(len(archives)-1,1))))
        else:
            ax.plot([i]*2, [0,probability], color = plt.cm.jet(n/float(max(len(archives)-1,1))))
ax.set_xlim(-10,260)
ax.legend()
plt.savefig('state_trace_contributions_by_eigenstatenr.pdf', dpi = 300)
