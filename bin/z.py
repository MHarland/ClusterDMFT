#!/usr/bin/env pytriqs
from cdmft.cdmft import CDmft
from pytriqs.gf.local import BlockGf
from cdmft.evaluation.quasiparticle import get_quasiparticle_residue
import sys

for n, arch in enumerate(sys.argv[1:]):
    c = CDmft(archive = arch)
    print arch[:-3] + ':'
    sigma_iw = c.load('Sigma_c_iw')
    for s in ['up', 'down']:
        z = [get_quasiparticle_residue(sigma_iw, n, s) for n in range(2, 6)]
        print 'Z_'+s+'(nr of matsubara frequencies of fit) = ', z
