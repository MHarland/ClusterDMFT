#!/usr/bin/env pytriqs
from cdmft.cdmft import CDmft
from cdmft.evaluation.observables import Observables
import sys

for arch in sys.argv[1:]:
    results = CDmft(archive = arch)
    g_iw = results.load('g_c_iw')
    obs = Observables(g_iw)
    print arch[:-3] + ':'
    mags = obs.local_magnetization()
    for orbSz in mags:
        print 'm_'+str(orbSz[0])+': '+str(orbSz[1])
    print 'm_total:'+str(obs.total_magnetization())
    print
