from itertools import izip
from numpy import array, exp, dot, pi, empty, identity
from pytriqs.gf.local import BlockGf, GfImFreq, iOmega_n, inverse

from .periodization import ClusterPeriodization
from ..mpiLists import scatter_list, allgather_list

# TODO make plots nice, what if cluster of lattice that is not fully trans inv?
# adjust plots for non-orthogonal reciprocal lattice vectors
# TODO FFT, izip from itertools, mpi

class SEPeriodization(ClusterPeriodization):
    """
    PRB 62 R9283
    """
    def set_sigma_lat(self, sigma, ssp):
        self.sigma_lat = _sigma_lat(sigma, ssp, self.bz_grid, self.spins)

    def set_g_lat(self, sigma_lat, mu):
        self.g_lat = _g_lat(sigma_lat, mu, self.eps, self.bz_grid, self.spins)

    def set_all(self, sigma, ssp, mu = 0):
        self.set_sigma_lat(sigma, ssp)
        self.set_g_lat(self.sigma_lat, mu)
        self.set_g_lat_loc(self.g_lat)
        self.set_tr_g_lat(self.g_lat)
        self.set_sigma_lat_loc(self.sigma_lat)
        self.set_tr_sigma_lat(self.sigma_lat)
        self.set_tr_g_lat_pade(self.g_lat)
        self.set_dos_loc(self.g_lat_loc)

# code with lists being parallelized, is faster than array views used in TRIQS sumk
def _sigma_lat(sigma, ssp, rbz_grid, spins):
    n_iwn = len(sigma[spins[0]].data[:, 0, 0])
    beta = sigma.beta
    n_k = len(rbz_grid)
    n_sites = len(ssp.values()[0])
    sigma_sl = [BlockGf(name_block_generator = [(s, GfImFreq(indices = range(n_sites), beta = beta, n_points = n_iwn)) for s in spins], name = '$\Sigma_{lat}$') for i in range(n_k)]
    sigTrans = dict()
    for r in ssp.keys():
        sigTrans[r] = empty([n_sites, n_sites], dtype = object)
        for a in range(n_sites):
            for b in range(n_sites):
                sigTrans[r][a,b] = ssp[r][a][b]
    sigma_sl_p = scatter_list(sigma_sl)
    for sig_k, k in izip(sigma_sl_p, scatter_list(rbz_grid)):
        for s, b in sig_k:
            for r, sigTrans_r in sigTrans.items():
                for a in range(n_sites):
                    for b in range(n_sites):
                        if type(sigTrans_r[a, b]) == tuple:
                            sig_k[s][a, b] += exp(complex(0, 2 * pi * dot(k, array(r)))) * sigma[s][sigTrans_r[a, b][0], sigTrans_r[a, b][1]]
    sigma_sl = allgather_list(sigma_sl_p)
    return sigma_sl

def _g_lat(sigma_lat, mu, eps, bz_grid, spins):
    n_k = len(bz_grid)
    n_bands = len(eps[0, :, :])
    g = [BlockGf(name_block_generator = [(s, sigma_lat[i][s]) for s in spins], name = '$G_{lat}$', make_copies = True) for i in range(n_k)]
    g_p = scatter_list(g)
    for g_k, sig_k, eps_k in izip(g_p, scatter_list(sigma_lat), scatter_list(eps)):
        for s, b in g_k:
            b << inverse((iOmega_n + mu) * identity(n_bands) - eps_k - sig_k[s])
    g = allgather_list(g_p)
    return g
