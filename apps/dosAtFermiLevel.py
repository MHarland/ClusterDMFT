#!/usr/bin/env pytriqs
from ClusterDMFT.cdmft import CDmft
from ClusterDMFT.evaluation.observables import Observables
import sys

for arch in sys.argv[1:]:
    results = CDmft(archive = arch)
    g_iw = results.load('g_c_iw')
    obs = Observables(g_iw)
    print arch[:-3] + ':'
    doss = obs.dosAtFermiLevel()
    for dos in doss:
        print 'n_'+str(dos[0][0])+'_'+str(dos[0][1])+': '+str(dos[1])
    print 'n_total: '+str(obs.totDosAtFermiLevel())
    print
