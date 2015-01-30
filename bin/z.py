#!/usr/bin/env pytriqs
from cdmft.cdmft import CDmft
from pytriqs.gf.local import BlockGf
from cdmft.evaluation.quasiparticle import get_quasiparticle_residue
import sys

for n, arch in enumerate(sys.argv[1:]):
    c = CDmft(archive = arch)
    print arch[:-3] + ':'
    sigma_iw = c.load('Sigma_c_iw')
    z = [get_quasiparticle_residue(sigma_iw, n) for n in range(2, 6)]
    print 'Z(nr of matsubara frequencies of fit) = ', z
