from pytriqs.gf.local import BlockGf, GfImFreq, iOmega_n, inverse
from numpy import array, exp, dot, pi, identity

from ..lattice.superlatticetools import dispersion as energy_dispersion
from .periodization import PeriodizationBase

# TODO make plots nice, what if cluster of lattice that is not fully trans inv?
# adjust plots for non-orthogonal reciprocal lattice vectors
# TODO FFT, izip from itertools, mpi

class Periodization(PeriodizationBase):
    """
    PRB 62 R9283
    uses superlattices quantities instead of lattice quantities
    """
    def set_sigma_lat(self, sigma, ssp):
        self.sigma_lat = _sigma_lat(sigma, self.bz_grid, ssp)

    def set_g_lat(self, sigma_lat, mu):
        self.g_lat = _g_lat(sigma_lat, mu, self.eps, self.bz_grid)

    def set_all(self, sigma, ssp, mu = 0):
        self.set_sigma_lat(sigma, ssp)
        self.set_g_lat(self.sigma_lat, mu)
        self.set_g_lat_loc(self.g_lat)
        self.set_tr_g_lat(self.g_lat)
        self.set_sigma_lat_loc(self.sigma_lat)
        self.set_tr_g_lat_pade(self.g_lat)

def _sigma_lat(sigma, rbz_grid, ssp):
    n_k = len(rbz_grid)
    zero = sigma.copy()
    zero.name = '$\Sigma_{lat}$'
    zero.zero()
    sigma_sl = [zero.copy() for k in range(n_k)]
    for k in range(n_k):
        for s, b in sigma_sl[k]:
            n_sites = len(b.data[0, :, :])
            for r in ssp.keys():
                sigma_r = array(ssp[r])
                for a in range(n_sites):
                    for b in range(n_sites):                        
                        if len(sigma_r[a, b]) == 2:
                            sigma_sl[k][s][a, b] += exp(complex(0, 2 * pi * dot(rbz_grid[k], array(r)))) * sigma[s][sigma_r[a, b][0], sigma_r[a, b][1]]
    return sigma_sl

def _g_lat(sigma_lat, mu, eps, bz_grid): # TODO only for full trans inv
    spins = ['up', 'down']
    n_kpts = len(bz_grid)
    n_bands = len(eps[0, :, :])
    g = [BlockGf(name_block_generator = [(s, GfImFreq(indices = range(n_bands), mesh = sigma_lat[0][spins[0]].mesh)) for s in spins], name = '$G_{lat}$') for i in range(n_kpts)]
    for s in spins:
        for k_ind in range(n_kpts):
            g[k_ind][s] << inverse(iOmega_n + mu * identity(n_bands) - eps[k_ind, :, :] - sigma_lat[k_ind][s])
    return g
