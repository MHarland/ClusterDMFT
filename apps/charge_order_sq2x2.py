#!/usr/bin/env pytriqs
from ClusterDMFT.cdmft import CDmft
from ClusterDMFT.evaluation.observables import Observables
import sys

for arch in sys.argv[1:]:
    results = CDmft(archive = arch)
    g_iw = results.load('g_c_iw')
    obs = Observables(g_iw)
    print arch[:-3] + ':'
    nco = obs.charge_order([[0, 3], [1, 2]])
    print 'n_CO: '+str(nco)
    print
