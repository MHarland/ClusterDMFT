#!/usr/bin/env pytriqs
from cdmft.cdmft import CDmft
from pytriqs.gf.local import BlockGf
from pytriqs.archive import HDFArchive
from cdmft.evaluation.quasiparticle import get_quasiparticle_residue
from cdmft.transformation.gf import g_sym
from numpy import array
import sys

for arch in sys.argv[1:]:
    c = CDmft(archive = arch)
    print arch[:-3] + ':'
    try:
        sigma_iw = c.load('Sigma_sym_iw')
    except KeyError:
        sigma_c_iw= c.load('Sigma_c_iw')
        archive = HDFArchive(arch, 'r')
        u = array(archive['parameters']['symmetry_transformation'])
        inds = archive['Results']['0']['sym_indices']
        del archive
        sigma_iw = g_sym(sigma_c_iw, u, inds)
        
    for s, b in sigma_iw:
        z = [get_quasiparticle_residue(sigma_iw, n, s, (0, 0)) for n in range(2, 7)]
        print 'Z_'+s+'(nr of matsubara frequencies of fit) = ', z
    print
