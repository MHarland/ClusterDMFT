#!/usr/bin/env pytriqs
from cdmft.cdmft import CDmft
from cdmft.evaluation.analytical_continuation import pade_tr as pade
from pytriqs.gf.local import BlockGf, GfImFreq, GfReFreq
from pytriqs.plot.mpl_interface import oplot
from numpy import pi
from matplotlib import pyplot as plt, cm
import sys

freq = int(sys.argv[1])
max_w = int(sys.argv[2])
n_graphs = len(sys.argv) -2

for n, arch in enumerate(sys.argv[3:len(sys.argv)]):
    x = CDmft(archive = arch)
    g = x.load('G_c_iw')
    g_w = pade(g, pade_n_omega_n = freq, pade_eta = 10**-10, dos_n_points = 1200, dos_window = (-max_w, max_w))
    oplot(g_w, RI = 'S', name = arch[0:-3], color = cm.jet(n /float(n_graphs)))
plt.savefig('dos'+str(freq)+'.png', dpi = 300)
plt.close()
print 'dos'+str(freq)+'.png ready'
del x
