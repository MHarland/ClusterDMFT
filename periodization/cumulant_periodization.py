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

    def set_m_lat(self, sigma, mu):
        self.m_lat = _cumulant_lat(sigma, self.bz_grid, 
                                   self.superlattice_basis, mu)

    def set_sigma_lat(self, m, mu):
        self.sigma_lat = _sigma_lat(m, self.bz_grid, mu)

    def set_g_lat(self, m):
        self.g_lat = _g_lat(m, energy_dispersion(self.lattice_vectors, 
                                                 self.lattice_basis, 
                                                 self.hopping, 
                                                 self.n_kpts), 
                            self.bz_grid)

    def set_all(self, sigma, mu = 0):
        self.set_m_lat(sigma, mu)
        self.set_sigma_lat(self.m_lat, mu)
        self.set_g_lat(self.m_lat)
        self.set_g_lat_loc(self.g_lat)
        self.set_tr_g_lat(self.g_lat)
        self.set_sigma_lat_loc(self.sigma_lat)
        self.set_tr_g_lat_pade(self.g_lat)


def _cumulant_lat(sigma, bz_grid, site_pos, mu):
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
        """
        if s == 'up': oplot(m[0, 0], x_window = (0, 40), marker = '+') # del
        m2 = GfImFreq(indices = range(n_sites), mesh = sigma.mesh) #
        m2.zero()
        tmp2 = GfImFreq(indices = range(n_sites), mesh = sigma.mesh)
        for n, iw_n in enumerate(sigma.mesh):
            tmp2 << iw_n + mu - sigma['up']
            m2.data[n, :, :] = inv(tmp2.data[n, :, :])
        if s == 'up': oplot(m2[0, 0], x_window = (0, 40), marker = '^') # del
        """
        for k_ind in range(n_kpts):
            k = bz_grid[k_ind]
            for i in range(n_sites):
                r_i = array(site_pos[i])
                for j in range(n_sites):
                    r_j = array(site_pos[j])
                    _temp = m_lat[k_ind][s].copy()
                    if i == j: m_lat[k_ind][s][0, 0] = _temp[0, 0] + m[s][i, j] * exp(complex(0, 2 * pi * dot(k, (r_i - r_j)))) /float(n_sites)
    del _temp
    """
    m_lat_loc = GfImFreq(indices = [0], mesh = sigma.mesh) # 
    for k_ind in range(n_kpts):
        tmp = m_lat_loc.copy()
        m_lat_loc << tmp + m_lat[k_ind]['up'] /float(16**2)
    oplot(m_lat_loc, x_window = (0, 40), marker = 'x') # del
    plt.show()
    """
    return m_lat

def _sigma_lat(m, bz_grid, mu):
    spins = ['up', 'down']
    n_kpts = len(bz_grid)
    sig = [BlockGf(name_block_generator = [(s, GfImFreq(indices = [0], mesh = m[0][spins[0]].mesh)) for s in spins], name = '$\Sigma_{lat}$') for i in range(n_kpts)]
    for s in spins:
        for k_ind in range(n_kpts):
            sig[k_ind][s] << iOmega_n + mu - inverse(m[k_ind][s])
    return sig

def _g_lat(m, eps, bz_grid): # TODO correct bandreduction
    spins = ['up', 'down']
    n_kpts = len(bz_grid)
    n_bands = len(eps)
    g = [BlockGf(name_block_generator = [(s, GfImFreq(indices = range(n_bands), mesh = m[0][spins[0]].mesh)) for s in spins], name = '$G_{lat}$') for i in range(n_kpts)]
    for s in spins:
        for b in range(n_bands):
            for k_ind in range(n_kpts):
                _temp = GfImFreq(indices = [0], mesh = m[0][spins[0]].mesh)
                _temp << inverse(m[k_ind][s])
                g[k_ind][s] << inverse(_temp - eps[0, k_ind])# TODO bands!
    return g
