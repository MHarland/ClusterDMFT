from numpy import pi, dot, ndarray, array, exp
from pytriqs.gf.local import iOmega_n, inverse
from pytriqs.plot.mpl_interface import oplot
from matplotlib import pyplot as plt

from .lattice.superlatticetools import dispersion, _init_k_sum
from .periodization.selfenergy_periodization import Periodization as SigmaPeriodization


class Cellular_DMFT(object):
    def __init__(self, lattice_vectors, clustersite_pos, hopping, n_kpts):
        self.k_sum = _init_k_sum(lattice_vectors, clustersite_pos, hopping, n_kpts)
        self.eps_rbz = self.k_sum.Hopping
        self.rbz_grid = self.k_sum.BZ_Points

    def g_local(self, sigma, mu):
        return self.k_sum(mu = mu, Sigma = sigma)

class PCDMFT_Li(object):
    """
    lattice_vectors: cartesian
    clustersite_pos: ?
    hopping: direct superlattice
    sigma_superlattice_periodization: direct superlattice : matrix(tuples/zeros) latter has type 3d array
    """
    def __init__(self, lattice_vectors, clustersite_pos, hopping, n_kpts, sigma_superlattice_periodization):
        k_sum = _init_k_sum(lattice_vectors, clustersite_pos, hopping, n_kpts)
        self.rbz_grid = k_sum.BZ_Points
        self.ssp = sigma_superlattice_periodization
        self.rbz_weights = k_sum.BZ_weights
        self.eps_rbz = k_sum.Hopping
        """
        t = self.eps_rbz[0].copy()
        for k in range(len(self.eps_rbz[:, 0, 0])):
            t += self.eps_rbz[k] / len(self.eps_rbz)**2
        print t
        """

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
        """
        s_loc = sigma.copy()
        for k in range(n_k):
            for s, b in s_loc:
                s_loc[s] += sigma_sl[k][s] * self.rbz_weights[k]
        oplot(sigma['up'], x_window=(0,100), marker='x', RI='I')
        oplot(s_loc['up'], x_window=(0,100), marker ='+', RI='I')
        plt.show()
        raise Exception('only a test')
        """
        g_loc.zero()
        for k in range(n_k):
            for s, b in g_loc:
                g_loc[s] += inverse(iOmega_n + mu - self.eps_rbz[k, :, :] - sigma_sl[k][s]) * self.rbz_weights[k]
        return g_loc

class MPCDMFT(object):
    """
    lattice_vectors: cartesian
    clustersite_pos: ?
    hopping: direct superlattice
    sigma_superlattice_periodization: direct superlattice : matrix(tuples/zeros) latter has type 3d array
    """
    def __init__(self, lattice_vectors, clustersite_pos, hopping, n_kpts, m_superlattice_periodization):
        k_sum = _init_k_sum(lattice_vectors, clustersite_pos, hopping, n_kpts)
        self.rbz_grid = k_sum.BZ_Points
        self.msp = m_superlattice_periodization
        self.rbz_weights = k_sum.BZ_weights
        self.eps_rbz = k_sum.Hopping
        """
        t = self.eps_rbz[0].copy()
        for k in range(len(self.eps_rbz[:, 0, 0])):
            t += self.eps_rbz[k] / len(self.eps_rbz)**2
        print t
        """

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
        """
        s_loc = sigma.copy()
        for k in range(n_k):
            for s, b in s_loc:
                s_loc[s] += sigma_sl[k][s] * self.rbz_weights[k]
        oplot(sigma['up'], x_window=(0,100), marker='x', RI='I')
        oplot(s_loc['up'], x_window=(0,100), marker ='+', RI='I')
        plt.show()
        raise Exception('only a test')
        """
        g_loc.zero()
        for k in range(n_k):
            for s, b in g_loc:
                g_loc[s] += inverse(inverse(m_sl[k][s]) - self.eps_rbz[k, :, :]) * self.rbz_weights[k]
        return g_loc


class PCDMFT_Kot(object):
    """
    lattice_vectors: cartesian
    lattice_basis: ?
    hopping: direct lattice coordinates
    clustersite_pos: direct lattice coordinates
    """
    def __init__(self, lattice_vectors, lattice_basis, hopping, n_kpts, clustersite_pos):
        k_sum = _init_k_sum(lattice_vectors, lattice_basis, hopping, n_kpts)
        self.bz_grid = k_sum.BZ_Points
        self.clustersite_pos = clustersite_pos
        self.bz_weights = k_sum.BZ_weights
        self.eps_bz = k_sum.Hopping
        self.lattice = SigmaPeriodization(lattice_vectors, lattice_basis, hopping, n_kpts, clustersite_pos)

    def g_local(self, sigma, mu):
        g_c = sigma.copy()
        g_c.zero()
        lattice = self.lattice
        lattice.set_sigma_lat(sigma)
        lattice.set_g_lat(lattice.get_sigma_lat(), mu)
        for i, r_i in enumerate(self.clustersite_pos):
            r_i = array(r_i)
            for j, r_j in enumerate(self.clustersite_pos):
                r_j = array(r_j)
                for l, k_l in enumerate(self.bz_grid):
                    for s, b in g_c:
                        b[i, j] += exp(complex(0, -2 * pi * dot(k_l, r_i - r_j))) * lattice.get_g_lat()[l][s][0, 0] * self.bz_weights[l]
        return g_c
