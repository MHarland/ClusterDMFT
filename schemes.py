from itertools import product
from numpy import pi, dot, ndarray, array, exp, zeros, identity, bmat
from pytriqs.gf.local import iOmega_n, inverse

from .lattice.superlatticetools import _init_k_sum
from .periodization.selfenergy_periodization import SEPeriodization
from .periodization.cumulant_periodization import MPeriodization

class Scheme(object):
    def __init__(self, scheme, cluster_lattice, cluster, t, n_kpts, periodization, blocks, nambu, *args, **kwargs):
        self.pretransf = lambda x: x
        self.pretransf_inv = lambda x: x
        self.nambu = nambu
        if scheme == 'cellular_dmft':
            self.selfconsistency = Cellular_DMFT(cluster_lattice, cluster, t, n_kpts)
        elif scheme == 'pcdmft':
            assert periodization, 'periodization required for PCDMFT'
            self.selfconsistency = PCDMFT(cluster_lattice, cluster, t, n_kpts, periodization, blocks)
        elif scheme == 'mpcdmft':
            assert periodization, 'periodization required for MPCDMFT'
            self.selfconsistency = MPCDMFT(cluster_lattice, cluster, t, n_kpts, periodization, blocks)

    def set_pretransf(self, transformation, inverse_transformation):
        """
        New Basis that the results will be expressed in, contrary to the other transformation that only takes place on the impurity.
        """
        self.pretransf = transformation
        self.pretransf_inv = inverse_transformation

    def apply_pretransf(self, g, call_by_value = False):
        """
        call_by_value = True makes a copy and returns it leaving the original g unchanged
        """
        if call_by_value:
            g = g.copy()
        return self.pretransf(g)

    def apply_pretransf_inv(self, g, call_by_value = True):
        if call_by_value:
            g = g.copy()
        return self.pretransf_inv(g)

    def g_local(self, sigma_c_iw, dmu, pretransf_inv = False):
        if self.nambu:
            return self._g_local_nambu(sigma_c_iw, dmu, pretransf_inv = False)
        else:
            return self._g_local(sigma_c_iw, dmu)

    def _g_local(self, sigma_c_iw, dmu):
        return self.selfconsistency.g_local(sigma_c_iw, dmu)

    def _g_local_nambu(self, sigma_c_iw, dmu, pretransf_inv = False):
        blocks = [ind for ind in sigma_c_iw.indices]
        d = len(sigma_c_iw[blocks[0]].data[0,:,:])
        field = [zeros([d, d])]
        for i in range(int(d/2), d):
            field[0][i,i] = 2 * dmu # maps mu -> -mu, taking care sign-change(s) in TRIQS sumk
        eps_nambu = lambda eps: bmat([[eps[:int(d/2),:int(d/2)],eps[:int(d/2),int(d/2):d]],
                                      [eps[int(d/2):d,:int(d/2)],-eps[int(d/2):d,int(d/2):d]]])
        if not pretransf_inv:
            return self.selfconsistency.g_local(sigma_c_iw, dmu, field = field, epsilon_hat = eps_nambu)
        else:
            return self.apply_pretransf_inv(self.selfconsistency.g_local(sigma_c_iw, dmu, field = field, epsilon_hat = eps_nambu), False) # call by ref due to copy in sumk

class Cellular_DMFT(Scheme):
    """
    RevModPhys.77.1027
    """
    def __init__(self, cluster_lattice, cluster, t, n_kpts, *args, **kwargs):
        self.k_sum = _init_k_sum(cluster_lattice, cluster, t, n_kpts)

    def g_local(self, sigma_c_iw, dmu, *args, **kwargs):
        return self.k_sum(mu = dmu, Sigma = sigma_c_iw, *args, **kwargs)

class PCDMFT(Scheme):
    """
    PhysRevB.62.R9283
    PhysRevB.69.205108
    """
    def __init__(self, cluster_lattice, cluster, t, n_kpts, periodization, blocks):
        self.lattice = SEPeriodization(cluster_lattice, cluster, t, n_kpts, blocks)
        self.periodization = periodization

    def g_local(self, sigma, mu): #TODO dmu etc
        self.lattice.set_sigma_lat(sigma, self.periodization)
        self.lattice.set_g_lat(self.lattice.get_sigma_lat(), mu)
        del self.lattice.sigma_lat
        self.lattice.set_g_lat_loc(self.lattice.get_g_lat())
        del self.lattice.g_lat
        return self.lattice.get_g_lat_loc()

class MPCDMFT(Scheme):
    """
    PCDMFT with cumulant periodization.
    """
    def __init__(self, cluster_lattice, cluster, t, n_kpts, periodization, blocks):
        self.lattice = MPeriodization(cluster_lattice, cluster, t, n_kpts, blocks)
        self.periodization = periodization

    def g_local(self, sigma, mu):
        self.lattice.set_m_lat(sigma, mu, self.periodization)
        self.lattice.set_g_lat(self.lattice.get_m_lat())
        del self.lattice.m_lat
        self.lattice.set_g_lat_loc(self.lattice.get_g_lat())
        del self.lattice.g_lat
        return self.lattice.get_g_lat_loc()
