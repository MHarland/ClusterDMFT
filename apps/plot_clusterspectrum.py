#!/usr/bin/env pytriqs
from ClusterDMFT.cdmft import CDmft
import sys, numpy
from matplotlib import pyplot as plt, cm

def get_index(x, listx):
    for i, xi in enumerate(listx):
        if numpy.allclose(x, xi):
            return i
    return None

def get_spectrum(eigensystems):
    degeneracies = list()
    energies = list()
    for subspace in eigensystems:
        energies_subspace = subspace[0]
        for e in energies_subspace:
            e_ind = get_index(e, energies)
            if e_ind == None:
                energies.append(e)
                degeneracies.append(1)
            else:
                degeneracies[e_ind] += 1
    return numpy.array([energies, degeneracies])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
archives = sys.argv[1:]
for n, arch in enumerate(sys.argv[1:]):
    c = CDmft(archive = arch)
    eigSys = c.load('eigensystems')
    spectrum = get_spectrum(eigSys)
    for e, d, i in zip(spectrum[0,:], spectrum[1,:], range(len(spectrum[0,:]))):
        if i == 0:
            ax.plot([e]*2, [0,d], label = arch[:-3], color = plt.cm.jet(n/float(max(len(archives)-1,1))))
        else:
            ax.plot([e]*2, [0,d], color = plt.cm.jet(n/float(max(len(archives)-1,1))))

ax.set_xlim(left = -.05)
ax.legend()
plt.savefig('clusterspectrum.pdf', dpi = 300)
