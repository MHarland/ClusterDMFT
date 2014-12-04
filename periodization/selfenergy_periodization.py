from pytriqs.gf.local import BlockGf, GfImFreq, iOmega_n, inverse
from numpy import array, exp, dot, pi

from ..lattice.superlatticetools import dispersion as energy_dispersion
from .periodization import PeriodizationBase

# TODO make plots nice, what if cluster of lattice that is not fully trans inv?
# adjust plots for non-orthogonal reciprocal lattice vectors
# TODO FFT, izip from itertools, mpi

class Periodization(PeriodizationBase):
    """
    Assumes full translation symmetry, 1 band.
    Tested for a square lattice.
    Some plotmethods are written for 2d-lattices only, the rest is supposed to be generic.
    """

    def set_sigma_lat(self, sigma):
        self.sigma_lat = _sigma_lat(sigma, self.bz_grid, self.superlattice_basis)

    def set_sigma_lat_loc(self, sigma_lat):
        self.sigma_lat_loc = _sigma_lat_loc(sigma_lat, self.bz_weights)

    def set_g_lat_loc(self, sigma_lat_loc, mu):
        self.g_lat_loc = _g_lat_loc(sigma_lat_loc, mu, energy_dispersion(self.lattice_vectors,
                                                                         self.lattice_basis, 
                                                                         self.hopping, 
                                                                         self.n_kpts), 
                                    self.bz_weights)
        

    def set_all(self, sigma, mu = 0):
        self.set_sigma_lat(sigma)
        self.set_sigma_lat_loc(self.sigma_lat)
        self.set_g_lat_loc(self.sigma_lat_loc, mu)

def _sigma_lat(sigma, bz_grid, site_pos):
    spins = ['up', 'down']
    d = len(bz_grid[0])
    n_kpts = len(bz_grid)
    n_sites = len(site_pos)
    if d == 2:
        assert len(bz_grid[:, 0]) == len(bz_grid[:, 1]), 'Fix GF datastructure!'
    if d == 3:
        assert len(bz_grid[:, 0]) == len(bz_grid[:, 1]) == len(bz_grid[:, 2]), 'Fix GF datastructure!'
    sigma_lat = [BlockGf(name_block_generator = [(s, GfImFreq(indices = [0], mesh = sigma.mesh)) for s in spins], name = '$\Sigma_{lat}$') for i in range(n_kpts)]
    for s in spins:
        for k_ind in range(n_kpts):
            k = bz_grid[k_ind]
            for i in range(n_sites):
                r_i = array(site_pos[i])
                for j in range(n_sites):
                    r_j = array(site_pos[j])
                    _temp = sigma_lat[k_ind][s].copy()
                    sigma_lat[k_ind][s][0, 0] << _temp[0, 0] + sigma[s][i, j] * exp(complex(0, 2 * pi * dot(k, (r_i - r_j)))) /float(n_sites**2)
    del _temp
    return sigma_lat

def _sigma_lat_loc(sigma_lat, bz_weights):
    n_kpts = len(bz_weights)
    sigma_lat_loc = sigma_lat[0].copy()
    sigma_lat_loc.name = '$\Sigma_{lat,loc}$'
    sigma_lat_loc.zero()
    for s, b in sigma_lat_loc:
        for k in range(n_kpts):
            sigma_lat_loc << sigma_lat_loc + sigma_lat[k] * bz_weights[k]
    return sigma_lat_loc

def _g_lat_loc(sigma_lat_loc, mu, eps, bz_weights):
    n_kpts = len(bz_weights)
    g_lat_loc = sigma_lat_loc.copy()
    g_lat_loc.name = '$G_{lat,loc}$'
    g_lat_loc.zero()
    for s, b in g0:
        for k in range(n_kpts):
            b << b + inverse(iOmega_n + mu - eps[0, k] - sigma_lat_loc[s]) * bz_weights[k]
    return g_lat_loc
