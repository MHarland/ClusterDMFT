from pytriqs.gf.local import BlockGf, GfImFreq, iOmega_n, inverse
from numpy import array, exp, dot, pi
from numpy.linalg import inv
from pytriqs.plot.mpl_interface import oplot
from matplotlib import pyplot as plt

from ..lattice.superlatticetools import dispersion as energy_dispersion
from .periodization import PeriodizationBase
#from transform import exact_inverse

# TODO make plots nice, what if cluster of lattice that is not fully trans inv?
# adjust plots for non-orthogonal reciprocal lattice vectors
# TODO FFT, izip from itertools, mpi

class Periodization(PeriodizationBase):
    """
    Phys.rev.B 74, 125110
    Phys.rev.B 85, 035102
    Assumes full translation symmetry, 1 band.
    Tested for a square lattice.
    Some plotmethods are written for 2d-lattices only, the rest is supposed to be generic.
    """

    def get_m_lat(self):
        return self.m_lat

    def set_m_lat(self, sigma, mu, ssp):
        self.m_lat = _cumulant_lat(sigma, mu, ssp, self.bz_grid)

    def set_sigma_lat(self, m, mu):
        self.sigma_lat = _sigma_lat(m, self.bz_grid, mu)

    def set_g_lat(self, m):
        self.g_lat = _g_lat(m, self.eps, self.bz_grid)

    def set_all(self, sigma, ssp, mu = 0):
        self.set_m_lat(sigma, mu, ssp)
        self.set_sigma_lat(self.m_lat, mu)
        self.set_g_lat(self.m_lat)
        self.set_g_lat_loc(self.g_lat)
        self.set_tr_g_lat(self.g_lat)
        self.set_sigma_lat_loc(self.sigma_lat)
        self.set_tr_g_lat_pade(self.g_lat)


def _cumulant_lat(sigma, mu, ssp, rbz_grid):
    spins = ['up', 'down']
    n_k = len(rbz_grid)
    n_sites = len(ssp.values()[0])
    m_sl = [BlockGf(name_block_generator = [(s, GfImFreq(indices = range(n_sites), mesh = sigma[spins[0]].mesh)) for s in spins], name = '$M_{lat}$') for i in range(n_k)]
    m_c = BlockGf(name_block_generator = [(s, GfImFreq(indices = range(len(sigma[spins[0]].data[0, :, :])), mesh = sigma[spins[0]].mesh)) for s in spins], name = '$M_C$')
    for s, b in m_c: b << inverse(iOmega_n + mu - sigma[s])
    for k in range(n_k):
        for s, b in m_sl[k]:
            for r in ssp.keys():
                m_r = array(ssp[r])
                for a in range(n_sites):
                    for b in range(n_sites):                        
                        if len(m_r[a, b]) == 2:
                            m_sl[k][s][a, b] += exp(complex(0, 2 * pi * dot(rbz_grid[k], array(r)))) * m_c[s][m_r[a, b][0], m_r[a, b][1]]
    return m_sl

def _sigma_lat(m, bz_grid, mu):
    spins = ['up', 'down']
    n_kpts = len(bz_grid)
    sig = [BlockGf(name_block_generator = [(s, m[i][s]) for s in spins], name = '$\Sigma_{lat}$', make_copies = True) for i in range(n_kpts)]
    for s in spins:
        for k_ind in range(n_kpts):
            sig[k_ind][s] << iOmega_n + mu - inverse(m[k_ind][s])
    return sig

def _g_lat(m, eps, bz_grid):
    spins = ['up', 'down']
    n_kpts = len(bz_grid)
    n_bands = len(eps[0, :, :])
    g = [BlockGf(name_block_generator = [(s, m[i][s]) for s in spins], name = '$G_{lat}$', make_copies = True) for i in range(n_kpts)]
    for s in spins:
        for k_ind in range(n_kpts):
            g[k_ind][s] << inverse(inverse(m[k_ind][s]) - eps[k_ind, :, :])
    return g
