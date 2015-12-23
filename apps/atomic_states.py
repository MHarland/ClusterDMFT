#!/usr/bin/env pytriqs
from ClusterDMFT.cdmft import CDmft
from ClusterDMFT.transformation.transformation import ClusterTransformationDMFT
from pytriqs.applications.impurity_solvers.cthyb import SolverCore as Solver
from pytriqs.operators import c as C, c_dag as C_dag
import sys
import numpy

def get_atomic_energies(eigensystems):
    energies = list()
    for subspace in impurity.eigensystems:
        for energy in subspace[0]:
            energies.append(energy)
    return numpy.array(energies)

def get_groundstatedegeneracy(energies, accuracy = 1e-12):
    counter = 0
    for energy in energies:
        if energy < accuracy:
            counter += 1
    return counter

for arch in sys.argv[1:]:
    print
    print arch[:-3]+':'
    results = CDmft(archive = arch)
    g_0_c_iw = results.load('g_0_c_iw')
    p = results.load('parameters')
    spins = [ind for ind in g_0_c_iw.indices]
    orbs = range(len(g_0_c_iw[spins[0]].data[0,:,:]))
    struct = [[s, orbs] for s in spins]
    hubbardSetter = ClusterTransformationDMFT(g_transf_struct = struct, g_loc = g_0_c_iw, transformation = numpy.identity(len(orbs)), beta = p['beta'], n_iw = p['n_iw'], spins = spins)
    hubbardSetter.set_hamiltonian(p['u_hubbard'], p['mu'], p['u_hubbard_non_loc'])
    impurity = Solver(beta = p['beta'], gf_struct = dict(struct), 
                      n_tau = p['n_tau'], n_iw = p['n_iw'], n_l = p['n_legendre'])
    impurity.G0_iw << g_0_c_iw
    impurity.solve(h_int = hubbardSetter.get_hamiltonian(), n_warmup_cycles = 0, n_cycles = 0, verbosity = 0)
    occupationOperators = dict([[s+'_'+str(i), C_dag(s, i) * C(s, i)] for s in spins for i in orbs])
    occupation_states = impurity.atomic_observables(occupationOperators)
    energies = get_atomic_energies(impurity.eigensystems)
    degeneracy = get_groundstatedegeneracy(energies)
    indices_groundstate = numpy.argsort(energies)[:degeneracy]
    print 'Groundstates:'
    for index in indices_groundstate:
        for orb, states in occupation_states.items():
            print orb+': '+str(states[index])
        print


