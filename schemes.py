from numpy import pi, dot, ndarray, array, exp
from pytriqs.gf.local import iOmega_n, inverse
from pytriqs.plot.mpl_interface import oplot
from matplotlib import pyplot as plt

from .lattice.superlatticetools import dispersion, _init_k_sum
from .periodization.selfenergy_periodization import Periodization as SigmaPeriodization

#todo implement independently scheme, periodization weights, periodization quantity, use additional ksum functionality
class Cellular_DMFT(object):
    def __init__(self, cluster_lattice, cluster, t, n_kpts, *args, **kwargs):
        self.k_sum = _init_k_sum(cluster_lattice, cluster, t, n_kpts)
        self.eps_rbz = self.k_sum.hopping
        self.rbz_grid = self.k_sum.bz_points

    def g_local(self, sigma, mu):
        return self.k_sum(mu = mu, Sigma = sigma)

class PCDMFT_Li(object):
    """
    lattice_vectors: cartesian
    clustersite_pos: ?
    hopping: direct superlattice
    periodization: direct superlattice coords, type is 3d array, ie matrix(tuples/zeros)
    """
    def __init__(self, cluster_lattice, cluster, t, n_kpts, periodization):
        k_sum = _init_k_sum(cluster_lattice, cluster, t, n_kpts)
        self.rbz_grid = k_sum.bz_points
        self.ssp = periodization
        self.rbz_weights = k_sum.bz_weights
        self.eps_rbz = k_sum.hopping

    def g_local(self, sigma, mu):
        n_k = len(self.rbz_grid)
        zero = sigma.copy()
        zero.zero()
        sigma_sl = [zero.copy() for k in range(n_k)]
        for k in range(n_k):
            for s, b in sigma_sl[k]:
                n_sites = len(b.data[0, :, :])
                for r in self.ssp.keys():
                    sigma_r = array(self.ssp[r])
                    for a in range(n_sites):
                        for b in range(n_sites):                        
                            if len(sigma_r[a, b]) == 2:
                                sigma_sl[k][s][a, b] += exp(complex(0, 2 * pi * dot(self.rbz_grid[k], array(r)))) * sigma[s][sigma_r[a, b][0], sigma_r[a, b][1]]
        g_loc = sigma.copy()
        g_loc.zero()
        for k in range(n_k):
            for s, b in g_loc:
                g_loc[s] += inverse(iOmega_n + mu - self.eps_rbz[k, :, :] - sigma_sl[k][s]) * self.rbz_weights[k]
        return g_loc

class MPCDMFT(object):
    def __init__(self, cluster_lattice, cluster, t, n_kpts, periodization):
        k_sum = _init_k_sum(cluster_lattice, cluster, t, n_kpts)
        self.rbz_grid = k_sum.bz_points
        self.msp = periodization
        self.rbz_weights = k_sum.bz_weights
        self.eps_rbz = k_sum.hopping

    def g_local(self, sigma, mu):
        n_k = len(self.rbz_grid)
        zero = sigma.copy()
        zero.zero()
        m_c = zero.copy()
        m_sl = [zero.copy() for k in range(n_k)]

        for s, b in m_c: b << inverse(iOmega_n + mu - sigma[s])
        for k in range(n_k):
            for s, b in m_sl[k]:
                n_sites = len(b.data[0, :, :])
                for r in self.msp.keys():
                    m_r = array(self.msp[r])
                    for a in range(n_sites):
                        for b in range(n_sites):                        
                            if len(m_r[a, b]) == 2:
                                m_sl[k][s][a, b] += exp(complex(0, 2 * pi * dot(self.rbz_grid[k], array(r)))) * m_c[s][m_r[a, b][0], m_r[a, b][1]]
        g_loc = sigma.copy()
        g_loc.zero()
        for k in range(n_k):
            for s, b in g_loc:
                g_loc[s] += inverse(inverse(m_sl[k][s]) - self.eps_rbz[k, :, :]) * self.rbz_weights[k]
        return g_loc

class PCDMFT_Kot(object):
    def __init__(self, lattice, lattice_basis, t, n_kpts, cluster):
        k_sum = _init_k_sum(lattice, lattice_basis, t, n_kpts)
        self.bz_grid = k_sum.bz_points
        self.cluster = cluster
        self.bz_weights = k_sum.bz_weights
        self.eps_bz = k_sum.hopping
        self.eps_rbz = self.eps_bz #here it is bz
        self.rbz_grid = self.bz_grid
        self.lattice = SigmaPeriodization(lattice, lattice_basis, t, n_kpts, cluster)

    def g_local(self, sigma, mu):
        g_c = sigma.copy()
        g_c.zero()
        lattice = self.lattice
        lattice.set_sigma_lat(sigma)
        lattice.set_g_lat(lattice.get_sigma_lat(), mu)
        for i, r_i in enumerate(self.cluster):
            r_i = array(r_i)
            for j, r_j in enumerate(self.cluster):
                r_j = array(r_j)
                for l, k_l in enumerate(self.bz_grid):
                    for s, b in g_c:
                        b[i, j] += exp(complex(0, -2 * pi * dot(k_l, r_i - r_j))) * lattice.get_g_lat()[l][s][0, 0] * self.bz_weights[l]
        return g_c

def get_scheme(parameters):
    p = parameters
    if p['scheme'] == 'cellular_dmft':
        scheme = Cellular_DMFT(p['cluster_lattice'], p['cluster'], p['t'], p['n_kpts'])
    elif p['scheme'] == 'pcdmft_li':
        assert 'periodization' in p, 'periodization required for PCDMFT_Li'
        scheme = PCDMFT_Li(p['cluster_lattice'], p['cluster'], p['t'], p['n_kpts'], p['periodization'])
    elif p['scheme'] == 'mpcdmft':
        assert 'periodization' in p, 'periodization required for PCDMFT_Li'
        scheme = MPCDMFT(p['cluster_lattice'], p['cluster'], p['t'], p['n_kpts'], p['periodization'])
    elif p['scheme'] == 'pcdmft_kot': # TODO written for full translation symmetry only, delete!
        assert 'hop_sublat' in p, 'hop_sublat required for PCDMFT_Kot'
        assert 'cluster_direct' in p, 'cluster_direct required for PCDMFT_Kot'
        scheme = PCDMFT_Kot(p['lattice_vectors'], [[0, 0, 0]], p['hop_sublat'], p['n_kpts'], p['cluster_direct'])
    return scheme

