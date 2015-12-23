#!/usr/bin/env pytriqs
from ClusterDMFT.cdmft import CDmft
import sys, numpy
from matplotlib import pyplot as plt, cm

def get_index(x, listx):
    for i, xi in enumerate(listx):
        if numpy.allclose(x, xi):
            return i
    return None

def get_spectrum(eigensystems, state_trace_contribs):
    dos = list()
    energies = list()
    stateCounter = 0
    for subspace in eigensystems:
        energies_subspace = subspace[0]
        indices_fockspace = subspace[2]
        for e, focknr in zip(energies_subspace, indices_fockspace):
            e_ind = get_index(e, energies)
            if e_ind == None:
                energies.append(e)
                dos.append(state_trace_contribs[stateCounter])
            else:
                dos[e_ind] += state_trace_contribs[stateCounter]
            stateCounter += 1
    return numpy.array([energies, dos])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
archives = sys.argv[1:]
linestyles = ['-', '--', '-.', ':'] * (len(archives)/4 + len(archives)%4)
for n, arch, ls in zip(range(len(archives)), sys.argv[1:], linestyles):
    c = CDmft(archive = arch)
    eigSys = c.load('eigensystems')
    probabilities = c.load('state_trace_contribs')
    spectrum = get_spectrum(eigSys, probabilities)
    for e, d, i in zip(spectrum[0,:], spectrum[1,:], range(len(spectrum[0,:]))):
        if i == 0:
            ax.plot([e]*2, [0,d], label = arch[:-3], color = plt.cm.jet(n/float(max(len(archives)-1,1))), ls = ls)
        else:
            ax.plot([e]*2, [0,d], color = plt.cm.jet(n/float(max(len(archives)-1,1))), ls = ls)

ax.set_xlim(left = -.05)
ax.legend()
plt.savefig('spectrum.pdf', dpi = 300)
