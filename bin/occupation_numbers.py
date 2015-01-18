#!/usr/bin/env pytriqs
from cdmft.cdmft import CDmft
from pytriqs.gf.local import BlockGf, GfImFreq, GfImTime
from numpy import array, savetxt
import sys

spins = ['up', 'down']

for n, arch in enumerate(sys.argv[1:]):
    occupation = list()
    disc = list()
    c = CDmft(archive = arch)
    g_iw = c.load('G_c_iw')

    for s, b in g_iw: orbitals = range(len(b.data[0, :, :]))
    g_tau = BlockGf(name_list = spins, block_list = [GfImTime(indices = orbitals, beta = g_iw.beta, n_points = c.parameters['n_tau']) for s in spins], make_copies = False)
    for s, b in g_tau: b.set_from_inverse_fourier(g_iw[s])

    for s, b in g_tau:
        for i in orbitals:
            occupation.append(-b.data[-1 ,i, i])
            disc.append(-b.data[0, i, i] - b.data[-1, i, i])
            assert 0.9 < disc[-1] and disc[-1] < 1.1, 'discontinuity is ' + str(disc[-1])
            print '<n_'+s+'_'+str(i)+str(i)+'> = ' + str(occupation[-1])
    print 'discontinuities: ', disc
    savetxt(arch[:-3] + '_occupation.txt', array(occupation))
    del c
