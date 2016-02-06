from pytriqs.gf.local import BlockGf, GfImFreq, delta, inverse, GfImTime
from pytriqs.utility.dichotomy import dichotomy
from .archive import ArchiveConnected
from .process_g import impose_site_symmetries, impose_paramagnetism, MixUpdate, impose_afm
from .transformation.nambuplaquette import g_spin_fullblock

class DMFTObjects(ArchiveConnected):
    """
    Initializes the objects that converge during the DMFT cycles
    """
    def __init__(self, archive, beta, n_iw, sigma_c_iw, dmu, blocks, blockstates, mix, *args, **kwargs):
        sigma_iw = sigma_c_iw
        super(DMFTObjects, self).__init__(archive)
        self.g_iw = BlockGf(name_block_generator = [(s, GfImFreq(indices = blockstates, beta = beta, n_points = n_iw, name = '$G_{c'+s+'}$')) for s in blocks], name = '$G_c$')
        self.sigma_iw = BlockGf(name_block_generator = [(s, GfImFreq(indices = blockstates, beta = beta, n_points = n_iw)) for s in blocks], name = '$\Sigma_c$')
        self.g_0_iw = BlockGf(name_block_generator = [(s, GfImFreq(indices = blockstates, beta = beta, n_points = n_iw)) for s in blocks], name = '$\\mathcal{G}$')
        if sigma_iw:
            self.sigma_iw << sigma_iw
        elif self.next_loop() > 0:
            self.sigma_iw = self.load('sigma_c_iw')
        else:
            self.sigma_iw.zero()
        if dmu or type(dmu) == int:
            self.dmu = dmu
        elif self.next_loop() > 0:
            self.dmu = self.load('dmu')
        else:
            self.dmu = 0
        if self.next_loop() > 0:
            self.mixing = MixUpdate(self.load('sigma_c_iw'), self.load('dmu'), mix)
        else:
            self.mixing = MixUpdate(self.sigma_iw, self.dmu, mix)

    def paramagnetic(self):
        """makes g, sigma and g0 paramagnetic by averaging"""
        for g in [self.g_iw, self.sigma_iw, self. g_0_iw]:
            g << impose_paramagnetism(g)

    def afm(self):
        """makes g, sigma and g0 paramagnetic by averaging"""
        for g in [self.g_iw, self.sigma_iw, self. g_0_iw]:
            g << impose_afm(g)

    def site_symmetric(self, site_symmetries):
        """makes g, sigma and g0 site symmetric by averaging according to site_symmetries"""
        for g in [self.g_iw, self.sigma_iw, self. g_0_iw]:
            g << impose_site_symmetries(g, site_symmetries)

    def mix(self):
        self.sigma_iw, self.dmu = self.mixing(self.sigma_iw, self.dmu)

    def find_dmu(self, scheme_obj, cluster_density, dmu_lim, dmu_step_lim, scheme, verbosity, *args, **kwargs):
        if cluster_density:
            if "nambu" in scheme:
                dens = lambda dmu : g_spin_fullblock(scheme_obj.g_local(self.sigma_iw, dmu)).total_density()
            else:
                dens = lambda dmu : scheme_obj.g_local(self.sigma_iw, dmu).total_density()
            dmu_old = self.dmu
            self.dmu, density0 = dichotomy(function = dens, x_init = self.dmu, 
                                           y_value = cluster_density, 
                                           precision_on_y = 0.001, delta_x = 0.5,
                                           max_loops = 1000, x_name = 'dmu', 
                                           y_name = 'cluster_density', verbosity = verbosity)
            if self.dmu == None: dmu = dmu_old
            if dmu_lim:
                if dmu > dmu_lim: self.dmu = dmu_lim
                if dmu < -dmu_lim: self.dmu = -dmu_lim
            if dmu_step_lim:
                if self.dmu - dmu_old > dmu_step_lim: dmu = dmu_old + dmu_step_lim
                if self.dmu - dmu_old < -dmu_step_lim: dmu = dmu_old - dmu_step_lim

    def make_g_0_iw_with_delta_tau_real(self, n_tau = 10000):
        delta_iw = delta(self.g_0_iw)
        delta_tau = self.get_delta_tau()
        # TODO delta_tau << delta_tau.real
        for s, b in delta_tau:
            for n, tau in enumerate(b.mesh):
                b.data[n,:,:] = b.data[n,:,:].real
        delta_iw_new = self.g_0_iw.copy()
        for s, b in delta_iw_new:
            b.set_from_fourier(delta_tau[s])
        g_0_inv = self.g_0_iw.copy()
        g_0_inv << inverse(self.g_0_iw)
        g_0_inv << g_0_inv + delta_iw - delta_iw_new
        self.g_0_iw << inverse(g_0_inv)
        
    def get_delta_tau(self, n_tau = 10000):
        delta_tau = BlockGf(name_list = [ind for ind in self.g_0_iw.indices], block_list = [GfImTime(beta = self.g_0_iw.beta, indices = [ind for ind in block.indices]) for blockname, block in self.g_0_iw], make_copies = False)
        for s in self.g_0_iw.indices:
            delta_tau[s].set_from_inverse_fourier(delta(self.g_0_iw[s]))
        return delta_tau

    def get_g_iw(self):
        return self.g_iw
    def set_g_iw(self, g):
        self.g_iw << g
    def get_sigma_iw(self):
        return self.sigma_iw
    def set_sigma_iw(self, g):
        self.sigma_iw << g
    def get_g_0_iw(self):
        return self.g_0_iw
    def set_g_0_iw(self, g):
        self.g_0_iw << g
    def get_dmu(self):
        return self.dmu
    def set_dmu(self, mu):
        self.dmu = mu
    def get_mixing(self):
        return self.mixing
    def get_dmft_objs(self):
        return self.g_0_iw, self.g_iw, self.sigma_iw, self.dmu
    def set_dmft_objs(self, g0, g, sigma, dmu = False):
        """sets G0, G and Sigma"""
        self.g_0_iw << g0
        self.g_iw << g
        self.sigma_iw << sigma
        if dmu: self.dmu = dmu

def t_local(t):
    for r, overlap_matrix in t.items():
        local = True
        for i in range(len(r)):
            if r[i] != 0:
                local = False
        if local:
            t_local = overlap_matrix
            break
    return t_local
