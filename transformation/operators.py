from numpy import array, identity, dot
from pytriqs.utility import mpi
from pytriqs.operators import c as C, c_dag as C_dag

from .u_matrix import CoulombTensor
from .other import sum_list, m_transform

def h_loc_sym(u_int, hop_loc, u, sym_indices, verbosity = 0):
    """
    transform operators and H_loc 
    express operators in site-basis by operators in new (energyeigen)-basis
    u_int: interaction
    hop_loc: local hopping i.e. t([0, ...])
    mu: chemical potential
    u: transformation-matrix
    sym_indices: indices after transformation
    """
    spins = ['up', 'down']
    dim = len(u)
    u = array(u)
    udag = u.T.conjugate()
    u_c = CoulombTensor(u_int, dim)
    u_c_sym = u_c.transform(u)
    assert dot(u, udag).all() == identity(len(u)).all(), 'transformation not unitary'
    c = dict()
    c_dag = dict()
    sites = range(dim)

    if verbosity > 0 and mpi.is_master_node():
        mpi.report('transformed local hopping is:')
        for i in range(dim):
            line = str()
            for j in range(dim):
                x = m_transform(hop_loc, u, i, j)
                if x >= 0:
                    line += '+' + str(x) + '  '
                else:
                    line += str(x) + '  '
            mpi.report(line)
        mpi.report('transformed Coulomb tensor is:')
        u_c_sym.show()

    h_loc = sum_list([sum_list([sum_list([_unblocked_c_dag_sym(s, i, sym_indices) * m_transform(hop_loc, u, i, j) * _unblocked_c_sym(s, j, sym_indices) for j in sites]) for i in sites]) for s in spins])
    for s1 in spins:
        for s2 in spins:
            for i in sites:
                for j in sites:
                    for k in sites:
                        for l in sites:
                            h_loc += u_c_sym[i, j, k, l, s1, s2] * _unblocked_c_dag_sym(s1, i, sym_indices) * _unblocked_c_dag_sym(s2, j, sym_indices) *  _unblocked_c_sym(s2, l, sym_indices) * _unblocked_c_sym(s1, k, sym_indices)
    if verbosity > 0 and mpi.is_master_node():
        mpi.report('H_loc:', h_loc)
    return h_loc

def _unblocked_c_sym(s, i, sym_indices, dag = False):
    ordered_keys = [sym_indices[ii][0] for ii in range(len(sym_indices))]
    j = 0

    for key in ordered_keys:
        for k in dict(sym_indices)[key]:
            if s in key:
                if i == j:
                    if dag:
                        return C_dag(key, k)
                    else:
                        return C(key, k)
                else:
                    j += 1

def _unblocked_c_dag_sym(s, i, sym_indices):
    return _unblocked_c_sym(s, i, sym_indices, dag = True)
