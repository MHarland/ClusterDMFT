from pytriqs.gf.local import BlockGf, GfImFreq
from pytriqs.utility.dichotomy import dichotomy
from .archive import ArchiveConnected
from .process_g import impose_site_symmetries, impose_paramagnetism, MixUpdate

class DMFTObjects(ArchiveConnected):
    """
    Initializes the objects that converge during the DMFT cycles
    """
    def __init__(self, archive, beta, n_iw, sigma_c_iw, dmu, spins, sites, mix, *args, **kwargs):
        sigma_iw = sigma_c_iw
        super(DMFTObjects, self).__init__(archive)
        g_init = GfImFreq(indices = sites, beta = beta, n_points = n_iw)
        self.g_iw = BlockGf(name_block_generator = [(s, GfImFreq(indices = sites, beta = beta, n_points = n_iw, name = '$G_{c'+s+'}$')) for s in spins], name = '$G_c$')
        self.sigma_iw = BlockGf(name_block_generator = [(s, g_init.copy()) for s in spins],
                                name = '$\Sigma_c$')
        self.g_0_iw = BlockGf(name_block_generator = [(s, g_init.copy()) for s in spins],
                              name = '$\\mathcal{G}$')
        del g_init
        if sigma_iw: 
            self.sigma_iw << sigma_iw
        elif self.next_loop() > 0:
            self.sigma_iw = self.load('sigma_c_iw')
        else:
            self.sigma_iw.zero()
        if dmu:
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

    def site_symmetric(self, site_symmetries):
        """makes g, sigma and g0 site symmetric by averaging according to site_symmetries"""
        for g in [self.g_iw, self.sigma_iw, self. g_0_iw]:
            g << impose_site_symmetries(g, site_symmetries)

    def mix(self):
        self.sigma_iw, self.dmu = self.mixing(self.sigma_iw, self.dmu)

    def find_dmu(self, scheme_obj, cluster_density, dmu_lim, dmu_step_lim, verbosity, *args, **kwargs):
        if cluster_density:
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
        return self.g_iw, self.sigma_iw, self.g_0_iw, self.dmu
    def set_dmft_objs(self, g0, g, sigma, dmu = False):
        """sets G0, G and Sigma"""
        self.g_0_iw << g0
        self.g_iw << g
        self.sigma_iw << sigma
        if dmu: self.dmu = dmu
