from itertools import izip
from numpy import array, exp, dot, pi, empty
from pytriqs.gf.local import BlockGf, GfImFreq, iOmega_n, inverse

from .periodization import ClusterPeriodization
from ..mpiLists import scatter_list, allgather_list

# TODO make plots nice, what if cluster of lattice that is not fully trans inv?
# adjust plots for non-orthogonal reciprocal lattice vectors
# TODO FFT, izip from itertools, mpi

class MPeriodization(ClusterPeriodization):
    """
    PhysRevB.74.125110
    PhysRevB.85.035102
    """
    def get_m_lat(self):
        return self.m_lat

    def set_m_lat(self, sigma, mu, ssp):
        self.m_lat = _cumulant_lat(sigma, mu, ssp, self.bz_grid, self.spins)

    def set_sigma_lat(self, m, mu):
        self.sigma_lat = _sigma_lat(m, self.bz_grid, mu, self.spins)

    def set_g_lat(self, m):
        self.g_lat = _g_lat(m, self.eps, self.bz_grid, self.spins)

    def set_all(self, sigma, ssp, mu = 0):
        self.set_m_lat(sigma, mu, ssp)
        self.set_sigma_lat(self.m_lat, mu)
        self.set_g_lat(self.m_lat)
        self.set_g_lat_loc(self.g_lat)
        self.set_tr_g_lat(self.g_lat)
        self.set_tr_sigma_lat(self.sigma_lat)
        self.set_sigma_lat_loc(self.sigma_lat)
        self.set_tr_g_lat_pade(self.g_lat)
        self.set_dos_loc(self.g_lat_loc)

# code with lists being parallelized, is faster than array views used in TRIQS sumk
def _cumulant_lat(sigma, mu, ssp, rbz_grid, spins):
    n_k = len(rbz_grid)
    n_sites = len(ssp.values()[0])
    m_sl = [BlockGf(name_block_generator = [(s, GfImFreq(indices = range(n_sites), mesh = sigma[spins[0]].mesh)) for s in spins], name = '$M_{lat}$') for i in range(n_k)]
    m_c = BlockGf(name_block_generator = [(s, GfImFreq(indices = range(len(sigma[spins[0]].data[0, :, :])), mesh = sigma[spins[0]].mesh)) for s in spins], name = '$M_C$')
    for s, b in m_c: b << inverse(iOmega_n + mu - sigma[s])
    mTrans = dict()
    for r in ssp.keys():
        mTrans[r] = empty([n_sites, n_sites], dtype = object)
        for a in range(n_sites):
            for b in range(n_sites):
                mTrans[r][a,b] = ssp[r][a][b]
    m_sl_p = scatter_list(m_sl)
    for m, k in izip(m_sl_p, scatter_list(rbz_grid)):
        for s, b in m:
            for r, mTrans_r in mTrans.items():
                for a in range(n_sites):
                    for b in range(n_sites):                  
                        if type(mTrans_r[a, b]) == tuple:
                            m[s][a, b] += exp(complex(0, 2 * pi * dot(k, array(r)))) * m_c[s][mTrans_r[a, b][0], mTrans_r[a, b][1]]
    m_sl = allgather_list(m_sl_p)
    return m_sl

def _sigma_lat(m, bz_grid, mu, spins):
    n_k = len(bz_grid)
    sig = [BlockGf(name_block_generator = [(s, m[0][s]) for s in spins], name = '$\Sigma_{lat}$', make_copies = True) for i in range(n_k)]
    sig_p = scatter_list(sig)
    for m_k, sig_k in izip(scatter_list(m), sig_p):
        for s in spins:
            sig_k[s] << iOmega_n + mu - inverse(m_k[s])
    sig = allgather_list(sig_p)
    return sig

def _g_lat(m, eps, bz_grid, spins):
    n_k = len(bz_grid)
    n_bands = len(eps[0, :, :])
    g = [BlockGf(name_block_generator = [(s, m[0][s]) for s in spins], name = '$G_{lat}$', make_copies = True) for i in range(n_k)]
    g_p = scatter_list(g)
    for g_k, m_k, eps_k in izip(g_p, scatter_list(m), scatter_list(eps)):
        for s in spins:
            g_k[s] << inverse(inverse(m_k[s]) - eps_k)
    g = allgather_list(g_p)
    return g


