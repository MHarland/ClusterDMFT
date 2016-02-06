from itertools import product
from numpy import pi, dot, ndarray, array, exp, zeros, identity, bmat
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

    def g_local(self, sigma_c_iw, dmu):
        return self.k_sum(mu = dmu, Sigma = sigma_c_iw)

class Cellular_DMFT_Nambu(object):
    """
    RevModPhys.77.1027
    """
    def __init__(self, cluster_lattice, cluster, t, n_kpts, *args, **kwargs):
        self.k_sum = _init_k_sum(cluster_lattice, cluster, t, n_kpts)

    def g_local(self, sigma_c_iw, dmu):
        blocks = [ind for ind in sigma_c_iw.indices]
        d = len(sigma_c_iw[blocks[0]].data[0,:,:])
        field = [zeros([d, d])]
        for i in range(int(d/2), d):
            field[0][i,i] = 2 * dmu # maps mu -> -mu, taking care sign-change(s) in TRIQS sumk
        eps_nambu = lambda eps: bmat([[eps[:int(d/2),:int(d/2)],eps[:int(d/2),int(d/2):d]],
                                      [eps[int(d/2):d,:int(d/2)],-eps[int(d/2):d,int(d/2):d]]])
        return self.k_sum(mu = dmu, Sigma = sigma_c_iw, field = field, epsilon_hat = eps_nambu)

class PCDMFT(object):
    """
    PhysRevB.62.R9283
    PhysRevB.69.205108
    """
    def __init__(self, cluster_lattice, cluster, t, n_kpts, periodization):
        self.lattice = SEPeriodization(cluster_lattice, cluster, t, n_kpts)
        self.periodization = periodization

    def g_local(self, sigma, mu): #TODO dmu etc
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
    elif p['scheme'] == 'cellular_dmft_nambu':
        scheme = Cellular_DMFT_Nambu(p['cluster_lattice'], p['cluster'], p['t'], p['n_kpts'])
    return scheme

