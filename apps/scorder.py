#!/usr/bin/env pytriqs
from ClusterDMFT.cdmft import CDmft
from ClusterDMFT.evaluation.observables import ObservablesTransfBasis
import sys

for arch in sys.argv[1:]:
    results = CDmft(archive = arch)
    g_iw = results.load('g_transf_iw')
    obs = ObservablesTransfBasis(g_iw)
    print arch[:-3] + ':'
    scos = obs.scorder_parameters()
    for k, sco in scos.items():
        print 'phi_'+k+' = '+str(sco[0])+' and '+str(sco[1])
    print
