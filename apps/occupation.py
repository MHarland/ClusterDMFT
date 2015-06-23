#!/usr/bin/env pytriqs
from cdmft.cdmft import CDmft
from cdmft.evaluation.observables import Observables
import sys

for arch in sys.argv[1:]:
    results = CDmft(archive = arch)
    g_iw = results.load('g_c_iw')
    obs = Observables(g_iw)
    print arch[:-3] + ':'
    occs = obs.occupation()
    for occ in occs:
        print 'n_'+str(occ[0][0])+'_'+str(occ[0][1])+': '+str(occ[1])
    print 'n_total: '+str(obs.total_occupation())
    print
