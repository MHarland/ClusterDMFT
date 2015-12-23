#!/usr/bin/env pytriqs
from ClusterDMFT.cdmft import CDmft
from pytriqs.gf.local import BlockGf, GfImFreq, GfImTime
from numpy import array, savetxt, pi
import sys

spins = ['up', 'down']

for n, arch in enumerate(sys.argv[1:]):
    dos = list()
    c = CDmft(archive = arch)
    g_iw = c.load('G_c_iw')

    print arch[:-3] + ':'

    for s, b in g_iw: orbitals = range(len(b.data[0, :, :]))
    g_tau = BlockGf(name_list = spins, block_list = [GfImTime(indices = orbitals, beta = g_iw.beta, n_points = c.parameters['n_tau']) for s in spins], make_copies = False)
    for s, b in g_tau: b.set_from_inverse_fourier(g_iw[s])
    betahalf = int(c.parameters['n_tau'] * .5)
    tau_mesh = [tau for n, tau in enumerate(g_tau['up'].mesh)]
    print 'assuming', tau_mesh[betahalf-1]-g_iw.beta, '>=',tau_mesh[betahalf]-g_iw.beta,'=<', tau_mesh[betahalf]-g_iw.beta

    for s, b in g_tau:
        for i in orbitals:
            dos.append(-b.beta * b.data[betahalf ,i, i] / pi)
            print '<n_'+s+'_'+str(i)+str(i)+'> = ' + str(dos[-1])
    savetxt(arch[:-3] + '_dos_estimate.txt', array(dos))
    del c
