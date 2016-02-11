from itertools import product
from numpy import array, identity, dot, zeros, sqrt
from numpy.linalg import inv as inverse
from pytriqs.gf.local import BlockGf, GfImFreq
import pytriqs
from pytriqs.operators import c as C, c_dag as C_dag

from .coulombinteraction import CoulombTensorHubbard, NNCoulombTensorHubbard
from .other import sum_list, m_transform

class NambuTransformation():
    """
    transforms the cluster objects required for the impurity problem within cdmft
    for the 1-orbital Hubbard model
    on Matsubara frequencies
    g_struct or gf_struct has an order, dict() it to make it TRIQS(CTHYB) compatible
    written and tested for unitary transformations
    """
    def __init__(self, g_transf_struct, transformation, beta, n_iw, blocks, blockstates, t, *args, **kwargs):
        self.transf_mat = array(transformation)
        self.blocks = blocks # initial blocks
        self.blockstates = blockstates # initial blockstates
        self.g_struct = g_transf_struct
        self.t_loc = array(t[(0,0)])[:len(self.transf_mat), :len(self.transf_mat)]
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

    def set_hamiltonian(self, u_hubbard, mu, u_hubbard_non_loc, *args, **kwargs):
        """
        transforms H_loc
        u_hubbard: Hubbard interaction
        mu: chemical potential that goes explicitly into the Hamiltonian (i.e. mu_DC)
        """
        spinspace = range(2)
        dim = len(self.transf_mat)
        u = self.transf_mat
        u_inv = inverse(self.transf_mat)
        u_c = CoulombTensorHubbard(-u_hubbard, dim, spinspace)
        u_c_transf = u_c.transform(u)
        assert dot(u, u_inv).all() == identity(len(u)).all(), 'transformation not unitary'
        mu_matrices = [(mu - u_hubbard) * identity(dim), -mu * identity(dim)]
        self.hamiltonian = 0
        self.hamiltonian -= sum_list([sum_list([sum_list([self._unblocked_c_dag(s, i) * m_transform(mu_matrix, u, i, j) * self._unblocked_c(s, j) for j in range(dim)]) for i in range(dim)]) for s, mu_matrix in zip(spinspace, mu_matrices)])
        for i, j, k, l, s1, s2 in product(*[range(dim)]*4 + [spinspace]*2):
            self.hamiltonian += u_c_transf[i, j, k, l, s1, s2] * self._unblocked_c_dag(s1, i) * self._unblocked_c_dag(s2, j) * self._unblocked_c(s2, l) * self._unblocked_c(s1, k)

    def get_hamiltonian(self):
        return self.hamiltonian

    def get_g_struct(self):
        return self.g_struct

    def _unblocked_c(self, s, i, dag = False):
        block = self.g_struct[i][0]
        if dag:
            return C_dag(block, s)
        else:
            return C(block, s)

    def _unblocked_c_dag(self, s, i):
        return self._unblocked_c(s, i, dag = True)

    def transform(self, g):
        return g_transf(g, self.transf_mat, self.g_struct, self.blocks)

    def backtransform(self, g):
        return g_c(g, self.transf_mat, self.g_struct, self.blocks, self.blockstates)

    def set_dmft_objs(self, g0, g, sigma, *args, **kwargs):
        """sets Weiss-Field, Green\'s function and self-energy at once"""
        self.set_g_iw(g)
        self.set_sigma_iw(sigma)
        self.set_g_0_iw(g0)
        
    def get_backtransformed_dmft_objs(self):
        """returns backtransformed Weiss-Field, Green\'s function and self-energy"""
        return self.backtransform(self.g_0_iw), self.backtransform(self.g_iw), self.backtransform(self.sigma_iw)

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

    def pretransformation(self):
        def g_nambu(g):
            """
            returns a transformed copy of the original greensfunction with one full block
            written for find_dmu
            """
            block_names = [bn for bn in g.indices]
            assert len(block_names) == 1, "nambu expects the up and down block to be merged into one block that can be transformed."
            block_name = block_names[0]
            assert type(g[block_name]) == GfImFreq, "G_nambu.total_density only for GfImFreq"
            spin_size = int(len(g[block_name].data[0,:,:])*.5)
            for i in range(spin_size, 2*spin_size):
                for j in range(spin_size, 2*spin_size):
                    g[block_name][i,j] << -1 * g[block_name][i,j].conjugate()
            return g
        return g_nambu

    def pretransformation_inverse(self):
        """
        inverse
        """
        return self.pretransformation()


def g_transf(g, transformation, transf_indices, blocks):
    assert len(blocks) == 1, 'nambu permits up/down blockstructures, choose one block!'
    initial_blockdim = len(g[blocks[0]].data[0,:,:])
    momenta = [transf_indices[i][0] for i  in range(len(transf_indices))]
    nambu_blocksize_momentum = [len(transf_indices[i][1]) for i  in range(len(transf_indices))]
    to_site = lambda initial_index: initial_index % int(initial_blockdim /2)
    inds_map_block = dict()
    inds_map_block[0] = range(0, int(initial_blockdim /2))
    inds_map_block[1] = range(int(initial_blockdim /2), initial_blockdim)
    inds_map_diag = dict()
    inds_map_diag[0] = range(0, int(initial_blockdim /2))
    u = array(transformation)
    u_inv = inverse(u)
    g_transf = BlockGf(name_block_generator = [(ind, GfImFreq(indices = dict(transf_indices)[ind], mesh = g[blocks[0]].mesh)) for ind in dict(transf_indices).keys()], make_copies = False)
    assert len(u)*2==initial_blockdim, 'blockstates must be twice as large as the site-space'
    for momentum_nr, momentum in enumerate(momenta):
        if nambu_blocksize_momentum[momentum_nr] == 1:
            inds_map = inds_map_diag
        else:
            inds_map = inds_map_block
        for i_nambu, j_nambu in product(inds_map.keys(), inds_map.keys()):
            g_transf[momentum][i_nambu, j_nambu] << sum_list([sum_list([u[momentum_nr, to_site(i)] * g[blocks[0]][i, j] * u_inv[to_site(j), momentum_nr] for i in inds_map[i_nambu]]) for j in inds_map[j_nambu]])
    return g_transf

def g_c(g, transformation, transf_indices, blocks, blockstates):
    u = array(transformation)
    u_inv = inverse(u)
    g_c = BlockGf(name_block_generator = [(s, GfImFreq(indices = blockstates, mesh = g[transf_indices[0][0]].mesh)) for s in blocks], make_copies = False)
    initial_blockdim = len(blockstates)
    momenta = [transf_indices[i][0] for i  in range(len(transf_indices))]
    to_site = lambda initial_index: initial_index %int(initial_blockdim /2)
    to_nambu = lambda initial_index: initial_index /int(initial_blockdim /2)
    nambu_size = dict([(transf_indices[i][0], len(transf_indices[i][1])) for i  in range(len(transf_indices))])
    gf = lambda mom, i, j: g[mom][to_nambu(i), to_nambu(j)] if to_nambu(i)<nambu_size[mom] and to_nambu(j)<nambu_size[mom] else 0
    momentum_nr = dict([(momentum, i) for i, momentum in enumerate(momenta)])
    for i, j in product(blockstates, blockstates):
        g_c[blocks[0]][i, j] << sum_list([u_inv[to_site(i), momentum_nr[momentum]] * gf(momentum, i, j) * u[momentum_nr[momentum], to_site(j)]  for momentum in momenta])
    return g_c

