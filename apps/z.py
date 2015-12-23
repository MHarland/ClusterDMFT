#!/usr/bin/env pytriqs
from ClusterDMFT.cdmft import CDmft
from pytriqs.gf.local import BlockGf
from ClusterDMFT.evaluation.quasiparticle import quasiparticle_residue
import sys

for n, arch in enumerate(sys.argv[1:]):
    c = CDmft(archive = arch)
    print arch[:-3] + ':'
    sigma_iw = c.load('sigma_c_iw')
    for s in ['up', 'down']:
        z = [quasiparticle_residue(sigma_iw, n, s, (0, 1)) for n in range(2, 7)]
        print 'Z_'+s+'(nr of matsubara frequencies of fit) = ', z
    print
