from pytriqs.gf.local import BlockGf, GfImFreq, iOmega_n, inverse
from numpy import array, exp, dot, pi
from numpy.linalg import inv

from ..lattice.superlatticetools import dispersion as energy_dispersion
from .cumulant_periodization import Periodization as MPeriodization

# TODO make plots nice, what if cluster of lattice that is not fully trans inv?
# adjust plots for non-orthogonal reciprocal lattice vectors
# TODO FFT, izip from itertools, mpi

class Periodization(MPeriodization):
    def __init__(self, ws_weights, *args, **kwargs):
        self.ws_weights = array(ws_weights)
        MPeriodization.__init__(self, *args, **kwargs)

    def set_m_lat(self, sigma, mu):
        self.m_lat = _cumulant_lat(sigma, self.bz_grid, self.superlattice_basis, mu, self.ws_weights)

def _cumulant_lat(sigma, bz_grid, site_pos, mu, ws_weights):
    spins = ['up', 'down']
    d = len(bz_grid[0])
    n_kpts = len(bz_grid)
    n_sites = len(site_pos)
    sites = range(n_sites)
    if d == 2:
        assert len(bz_grid[:, 0]) == len(bz_grid[:, 1]), 'Fix GF datastructure!'
    if d == 3:
        assert len(bz_grid[:, 0]) == len(bz_grid[:, 1]) == len(bz_grid[:, 2]), 'Fix GF datastructure!'
    m_lat = [BlockGf(name_block_generator = [(s, GfImFreq(indices = [0], mesh = sigma.mesh)) for s in spins], name = '$M_{lat}$') for i in range(n_kpts)] # TODO for not full sym
    m = BlockGf(name_block_generator = [(s, GfImFreq(indices = sites, mesh = sigma.mesh)) for s in spins], name = '$M_C$')
    m_inv = BlockGf(name_block_generator = [(s, GfImFreq(indices = sites, mesh = sigma.mesh)) for s in spins], name = '$M_C^{-1}$')
    for s, b in m_inv: b << iOmega_n + mu - sigma[s]
    m << inverse(m_inv)
    for s in spins:
        for k_ind in range(n_kpts):
            k = bz_grid[k_ind]
            for i in range(n_sites):
                r_i = array(site_pos[i])
                for j in range(n_sites):
                    r_j = array(site_pos[j])
                    m_lat[k_ind][s][0, 0] += ws_weights[i, j] * m[s][i, j] * exp(complex(0, 2 * pi * dot(k, (r_i - r_j))))
    return m_lat
