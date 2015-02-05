#!/usr/bin/env pytriqs
from cdmft.cdmft import CDmft
from cdmft.evaluation.analytical_continuation import pade_tr as pade
from pytriqs.gf.local import BlockGf, GfImFreq, GfReFreq
from pytriqs.plot.mpl_interface import oplot
from numpy import pi
from matplotlib import pyplot as plt, cm
import sys

n_start = int(sys.argv[1])
n_stop = int(sys.argv[2])
n_step = int(sys.argv[3])
max_w = int(sys.argv[4])

for arch in sys.argv[5:len(sys.argv)]:
    x = CDmft(archive = arch)
    g = x.load('G_c_iw')
    for n in range(n_start, n_stop, n_step):
        g_w = pade(g, pade_n_omega_n = n, pade_eta = 10**-10, dos_n_points = 1200, dos_window = (-max_w, max_w))
        oplot(g_w, RI = 'S', name = str(n), color = cm.jet((n - n_start) * n_step/float(n_stop-n_start)))
    filename = 'dos_' + arch[0:-3] + '_' + str(n_start) + str(n_stop) + str(n_step) + '.png'
    plt.savefig(filename)
    plt.close()
    print filename + ' ready'
    del x

