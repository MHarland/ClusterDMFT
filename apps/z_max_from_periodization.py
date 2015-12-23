#!/usr/bin/env pytriqs
from ClusterDMFT.periodization.periodization import PeriodizationBase as Periodization
from pytriqs.gf.local import BlockGf
from ClusterDMFT.evaluation.quasiparticle import get_quasiparticle_residue
from numpy import array
from itertools import izip
import sys

for n, arch in enumerate(sys.argv[1:]):
    c = Periodization(archive = arch)
    print arch[:-3] + ':'
    sigma_iw = c.get_sigma_lat()
    mesh = c.get_bz_grid()
    z = list()
    for k, sig in izip(mesh, sigma_iw):
        for s in ['up', 'down']:
            z.append(get_quasiparticle_residue(sig, n, s, (0,0)) for n in range(2, 3))
    z=array(z)
    print 'Z_'+s+'(nr of matsubara frequencies of fit) = ', z.max()
    print 'k = ', mesh[z.argmax()]
    print
