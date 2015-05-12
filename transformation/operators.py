from numpy import array, identity, dot
from pytriqs.utility import mpi
from pytriqs.operators import c as C, c_dag as C_dag

from .u_matrix import CoulombTensor, NNCoulombTensor
from .other import sum_list, m_transform

def h_loc_sym(u_int, mu, u, sym_indices, u_nn_int = False):
    """
    transform operators and H_loc 
    express operators in site-basis by operators in new (energyeigen)-basis
    u_int: interaction
    hop_loc: local hopping i.e. t([0, ...])
    mu: chemical potential
    u: transformation-matrix in site-space
    sym_indices: indices after transformation
    """
    spins = ['up', 'down']
    dim = len(u)
    u = array(u)
    udag = u.T.conjugate()
    u_c = CoulombTensor(u_int, dim)
    u_c_sym = u_c.transform(u)
    if u_nn_int:
        u_nn_c = NNCoulombTensor(u_nn_int, dim)
        u_nn_c_sym = u_nn_c.transform(u)
    assert dot(u, udag).all() == identity(len(u)).all(), 'transformation not unitary'
    c = dict()
    c_dag = dict()
    sites = range(dim)

    mu_matrix = - mu * identity(dim)
    h_loc = sum_list([sum_list([sum_list([_unblocked_c_dag_sym(s, i, sym_indices) * m_transform(mu_matrix, u, i, j) * _unblocked_c_sym(s, j, sym_indices) for j in sites]) for i in sites]) for s in spins])

    for s1 in spins:
        for s2 in spins:
            for i in sites:
                for j in sites:
                    for k in sites:
                        for l in sites:
                            h_loc += u_c_sym[i, j, k, l, s1, s2] * _unblocked_c_dag_sym(s1, i, sym_indices) * _unblocked_c_dag_sym(s2, j, sym_indices) *  _unblocked_c_sym(s2, l, sym_indices) * _unblocked_c_sym(s1, k, sym_indices)
    if u_nn_int:
        for s1 in spins:
            for s2 in spins:
                for i in sites:
                    for j in sites:
                        for k in sites:
                            for l in sites:
                                h_loc += u_nn_c_sym[i, j, k, l, s1, s2] * _unblocked_c_dag_sym(s1, i, sym_indices) * _unblocked_c_dag_sym(s2, j, sym_indices) *  _unblocked_c_sym(s2, l, sym_indices) * _unblocked_c_sym(s1, k, sym_indices)

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
