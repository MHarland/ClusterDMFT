#!/usr/bin/env pytriqs
from ClusterDMFT.periodization.periodization import PeriodizationBase as Periodization
from pytriqs.gf.local import BlockGf
from ClusterDMFT.evaluation.quasiparticle import get_quasiparticle_residue
import sys

for n, arch in enumerate(sys.argv[1:]):
    c = Periodization(archive = arch)
    print arch[:-3] + ':'
    sigma_iw = c.get_sigma_lat_loc()
    for s in ['up', 'down']:
        z = [get_quasiparticle_residue(sigma_iw, n, s, (0,0)) for n in range(2, 6)]
        print 'Z_'+s+'(nr of matsubara frequencies of fit) = ', z
