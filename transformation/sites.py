from itertools import product
from numpy import array, identity, dot, zeros
from numpy.linalg import inv as inverse
from pytriqs.gf.local import BlockGf, GfImFreq
from pytriqs.operators import c as C, c_dag as C_dag

from .coulombinteraction import CoulombTensorHubbard, NNCoulombTensorHubbard
from .other import sum_list, m_transform

class ClustersiteTransformation():
    """
    transforms the cluster objects required for the impurity problem within cdmft
    for the 1-orbital Hubbard model
    on Matsubara frequencies
    g_struct or gf_struct has an order, dict() it to make it TRIQS(CTHYB) compatible
    written and tested for unitary transformations
    """
    def __init__(self, g_transf_struct, transformation, beta, n_iw, blocks, *args, **kwargs):
        self.transf_mat = array(transformation)
        self.blocks = blocks
        if g_transf_struct:
            self.g_struct = g_transf_struct
        else:
            assert 'g_loc' in kwargs.keys(), 'ClusterTransformation needs either g_transf_struct or g_loc to init'
            g_loc = kwargs['g_loc']
            self.g_struct = transf_indices(g_loc, transformation, self.blocks)
        name_blocks = [(ind, GfImFreq(indices = dict(self.g_struct)[ind], 
                                      beta = beta, 
                                      n_points = n_iw)
                        ) for ind in dict(self.g_struct).keys()]
        self.g_iw = BlockGf(name_block_generator = name_blocks,
                            make_copies = True,
                            name = '$\\tilde{G}_c$')
        self.g_0_iw = BlockGf(name_block_generator = name_blocks,
                              make_copies = True,
                              name = '$\\tilde{\\mathrm{G}}$')
        self.sigma_iw = BlockGf(name_block_generator = name_blocks,
                                make_copies = True,
                                name = '$\\tilde{\\Sigma}_c$')

    def set_hamiltonian(self, u_hubbard, mu, u_hubbard_non_loc, blocks, *args, **kwargs):
        """
        transforms H_loc
        express operators in site-basis by operators in new basis
        u: Hubbard interaction, can also be non-local(matrix-valued) within the cluster
        mu: chemical potential that goes explicitly into the Hamiltonian (i.e. mu_DC)
        """
        dim = len(self.transf_mat)
        u = self.transf_mat
        u_inv = inverse(self.transf_mat)
        u_c = CoulombTensorHubbard(u_hubbard, dim, blocks)
        u_c_transf = u_c.transform(u)
        if u_hubbard_non_loc:
            u_c_nl = NNCoulombTensorHubbard(u_hubbard_non_loc, dim, blocks)
            u_c_nl_transf = u_c_nl.transform(u)
        assert dot(u, u_inv).all() == identity(len(u)).all(), 'transformation not unitary'
        sites = range(dim)
        mu_matrix = - mu * identity(dim)
        self.hamiltonian = sum_list([sum_list([sum_list([self._unblocked_c_dag(s, i) * m_transform(mu_matrix, u, i, j) * self._unblocked_c(s, j) for j in sites]) for i in sites]) for s in self.blocks])
        for i, j, k, l, s1, s2 in product(*[sites]*4 + [self.blocks]*2):
            self.hamiltonian += u_c_transf[i, j, k, l, s1, s2] * self._unblocked_c_dag(s1, i) * self._unblocked_c_dag(s2, j) *  self._unblocked_c(s2, l) * self._unblocked_c(s1, k)
        if u_hubbard_non_loc:
            for i, j, k, l, s1, s2 in product(*[sites]*4 + [self.blocks]*2):
                self.hamiltonian += u_c_nl_transf[i, j, k, l, s1, s2] * self._unblocked_c_dag(s1, i) * self._unblocked_c_dag(s2, j) *  self._unblocked_c(s2, l) * self._unblocked_c(s1, k)

    def get_hamiltonian(self):
        return self.hamiltonian

    def get_g_struct(self):
        return self.g_struct

    def _unblocked_c(self, s, i, dag = False):
        ordered_keys = [self.g_struct[ii][0] for ii in range(len(self.g_struct))]
        j = 0
        for key in ordered_keys:
            for k in dict(self.g_struct)[key]:
                if s in key:
                    if i == j:
                        if dag:
                            return C_dag(key, k)
                        else:
                            return C(key, k)
                    else:
                        j += 1

    def _unblocked_c_dag(self, s, i):
        return self._unblocked_c(s, i, dag = True)

    def transform(self, g):
        return g_transf(g, self.transf_mat, self.g_struct, self.blocks)

    def backtransform(self, g):
        g_c(g, self.transf_mat, self.g_struct, self.blocks)

    def set_dmft_objs(self, g0, g, sigma):
        """sets Weiss-Field, Green\'s function and self-energy at once"""
        self.set_g_0_iw(g0)
        self.set_g_iw(g)
        self.set_sigma_iw(sigma)

    def get_backtransformed_dmft_objs(self):
        """returns backtransformed Weiss-Field, Green\'s function and self-energy"""
        return g_c(self.g_0_iw, self.transf_mat, self.g_struct, self.blocks), g_c(self.g_iw, self.transf_mat, self.g_struct, self.blocks), g_c(self.sigma_iw, self.transf_mat, self.g_struct, self.blocks)

    def set_g_0_iw(self, g):
        if [ind for ind in g.indices] == dict(self.g_struct).keys():
            self.g_0_iw << g
        else:
            self.g_0_iw << g_transf(g, self.transf_mat, self.g_struct, self.blocks)
    def get_g_0_iw(self):
        return self.g_0_iw
    def set_g_iw(self, g):
        if [ind for ind in g.indices] == dict(self.g_struct).keys():
            self.g_iw << g
        else:
            self.g_iw << g_transf(g, self.transf_mat, self.g_struct, self.blocks)
    def get_g_iw(self):
        return self.g_iw
    def set_sigma_iw(self, g):
        if [ind for ind in g.indices] == dict(self.g_struct).keys():
            self.sigma_iw << g
        else:
            self.sigma_iw << g_transf(g, self.transf_mat, self.g_struct, self.blocks)
    def get_sigma_iw(self):
        return self.sigma_iw
    

def transf_indices(g_c, transformation, blocks, almost_zero = 10e-9):
    """
    Finds the new blockstructure of G_loc using the unitary transformation u.
    transformation is a matrix over the cluster sites
    """
    n_sites = len(transformation)
    sites = range(n_sites)
    g_block_structure = zeros([n_sites, n_sites], int)
    u = array(transformation)
    u_inv = inverse(u)
    blockdiag_len = list()
    _blockdiag_len = 0
    for s, b in g_c:
        for i, l, m in product(sites, sites, range(len(b.data[:,0,0]))):
            if abs(sum_list([sum_list([u[i, j] * b.data[m, j, k] * u_inv[k, l] for j in sites]) for k in sites])) > almost_zero:
                g_block_structure[i, l] = 1
    for i in sites:
        block_is_zero = True
        for j in range(i, n_sites, 1):
            for k in range(0, i + 1, 1):
                if i != j and (g_block_structure[i - k, j] != 0 or g_block_structure[j, i - k] != 0):
                    block_is_zero = False
        if block_is_zero:
            blockdiag_len.append(_blockdiag_len)
            _blockdiag_len = 0
        else:
            _blockdiag_len += 1
    return [(str(i) + '-' + s, range(blockdiag_len[i] + 1)) for s in blocks for i in range(len(blockdiag_len))]

def g_transf(g, transformation, transf_indices, blocks):
    u = array(transformation)
    u_inv = inverse(u)
    g_transf = BlockGf(name_block_generator = [(ind, GfImFreq(indices = dict(transf_indices)[ind], mesh = g[blocks[0]].mesh)) for ind in dict(transf_indices).keys()], make_copies = False)
    for ss in blocks:
        i = 0
        for s in [transf_indices[ii][0] for ii in range(len(transf_indices))]:
            if ss in s:
                for ib in dict(transf_indices)[s]:
                    for jb in dict(transf_indices)[s]:
                        g_transf[s][ib, jb] << sum_list([sum_list([u[i + ib, k] * g[ss][k, l] * u_inv[l, i + jb] for k in range(len(u))]) for l in range(len(u))])
                i += 1
    return g_transf

def g_c(g, transformation, transf_indices, blocks):
    u = array(transformation)
    u_inv = inverse(u)
    sites = range(len(u))
    g_c = BlockGf(name_block_generator = [(s, GfImFreq(indices = sites, mesh = g[transf_indices[0][0]].mesh)) for s in blocks], make_copies = False)
    for s, i, j in product(blocks, sites, sites):
        g_c[s][i, j] << sum_list([sum_list([u_inv[i, k] * _unblocked_g_transf(g, s, k, l, transf_indices) * u[l, j]  for k in sites]) for l in sites])
    return g_c

def _unblocked_g_transf(g, block, row, col, transf_indices):
    i0 = row if row <= col else col
    i1 = row if row > col else col
    ordered_keys = [transf_indices[ii][0] for ii in range(len(transf_indices))]
    j = 0
    for key in ordered_keys:
        if block in key:
            for k in dict(transf_indices)[key]:
                if i0 == j:
                    (block, ind1) = (key, k)
                j += 1
    if len(dict(transf_indices)[block]) > ind1 + i1 - i0:
        if row <= col:
            return g[block][ind1, ind1 + i1 - i0]
        else:
            return g[block][ind1 + i1 - i0, ind1]
    else:
        return 0
