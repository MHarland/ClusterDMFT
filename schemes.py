from numpy import pi, dot, ndarray, array, exp
from pytriqs.gf.local import iOmega_n, inverse

from .lattice.superlatticetools import _init_k_sum
from .periodization.selfenergy_periodization import SEPeriodization
from .periodization.cumulant_periodization import MPeriodization

class Cellular_DMFT(object):
    """
    RevModPhys.77.1027
    """
    def __init__(self, cluster_lattice, cluster, t, n_kpts, *args, **kwargs):
        self.k_sum = _init_k_sum(cluster_lattice, cluster, t, n_kpts)

    def g_local(self, sigma, mu):
        return self.k_sum(mu = mu, Sigma = sigma)

class PCDMFT(object):
    """
    PhysRevB.62.R9283
    PhysRevB.69.205108
    """
    def __init__(self, cluster_lattice, cluster, t, n_kpts, periodization):
        self.lattice = SEPeriodization(cluster_lattice, cluster, t, n_kpts)
        self.periodization = periodization

    def g_local(self, sigma, mu):
        self.lattice.set_sigma_lat(sigma, self.periodization)
        self.lattice.set_g_lat(self.lattice.get_sigma_lat(), mu)
        del self.lattice.sigma_lat
        self.lattice.set_g_lat_loc(self.lattice.get_g_lat())
        del self.lattice.g_lat
        return self.lattice.get_g_lat_loc()

class MPCDMFT(object):
    """
    PCDMFT with cumulant periodization.
    """
    def __init__(self, cluster_lattice, cluster, t, n_kpts, periodization):
        self.lattice = MPeriodization(cluster_lattice, cluster, t, n_kpts)
        self.periodization = periodization

    def g_local(self, sigma, mu):
        self.lattice.set_m_lat(sigma, mu, self.periodization)
        self.lattice.set_g_lat(self.lattice.get_m_lat())
        del self.lattice.m_lat
        self.lattice.set_g_lat_loc(self.lattice.get_g_lat())
        del self.lattice.g_lat
        return self.lattice.get_g_lat_loc()

def get_scheme(parameters):
    p = parameters
    if p['scheme'] == 'cellular_dmft':
        scheme = Cellular_DMFT(p['cluster_lattice'], p['cluster'], p['t'], p['n_kpts'])
    elif p['scheme'] == 'pcdmft':
        assert 'periodization' in p, 'periodization required for PCDMFT'
        scheme = PCDMFT(p['cluster_lattice'], p['cluster'], p['t'], p['n_kpts'], p['periodization'])
    elif p['scheme'] == 'mpcdmft':
        assert 'periodization' in p, 'periodization required for MPCDMFT'
        scheme = MPCDMFT(p['cluster_lattice'], p['cluster'], p['t'], p['n_kpts'], p['periodization'])
    return scheme

