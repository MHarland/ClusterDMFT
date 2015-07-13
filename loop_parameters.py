from numpy import ndarray, array, zeros, identity
from .utility import get_dim, get_n_sites

class CleanLoopParameters(dict):
    """
    basically the parameters dictionary, but with additional functions that clean-up the parameters
    """
    _obligatory = ['archive', 'cluster_lattice', 'cluster', 'u_hubbard', 't', 'beta', 'n_cycles', 'mu', 'n_kpts']
    _cthyb_keys = ['n_cycles', 'length_cycle', 'n_warmup_cycles', 'random_seed', 'random_name', 'max_time', 'verbosity', 'use_trace_estimator', 'measure_g_tau', 'measure_g_l', 'measure_pert_order', 'move_shift', 'move_double']
    _replenishing_parameters = {'cluster_density': False,
                                'verbosity': 1,
                                'scheme': 'cellular_dmft',
                                'transformation': False,
                                'density': False,
                                'g_transf_struct': False,
                                'sigma_c_iw': False,
                                'dmu': False,
                                'spins': ['up', 'down'],
                                'mu': 0,
                                'u_hubbard_non_loc': 0,
                                'n_iw': 1025,
                                'n_tau': 10001,
                                'mix': 1,
                                'dmu_lim': False,
                                'dmu_step_lim': False,
                                'fit_tail': True,
                                'tail_start': 50,
                                'n_legendre': 50,
                                'site_symmetries': False,
                                'impose_paramagnetism': False,
                                'g_transf_struct': False,
                                'ext_field': False,
                                'length_cycle': 50,
                                'n_warmup_cycles': 5000,
                                'random_seed': False,
                                'random_name': False,
                                'max_time': -1,
                                'use_trace_estimator': False,
                                'measure_g_tau': True,
                                'measure_g_l': True,
                                'measure_pert_order': True,
                                'move_shift': True,
                                'move_double': True
                                }

    def __init__(self, *args, **kwargs):
        super(CleanLoopParameters, self).__init__(*args, **kwargs)
        if CleanLoopParameters._missing_dmft_parameters(self):
            self.update(CleanLoopParameters._replenish(CleanLoopParameters._missing_dmft_parameters(self)))
        self.update(CleanLoopParameters._adjust(self))
        assert not CleanLoopParameters._missing_dmft_parameters(self), str(CleanLoopParameters._missing_dmft_parameters(self)) + 'are missing'
        
    def get_cthyb_parameters(self):
        cthyb = dict()
        for key in CleanLoopParameters._cthyb_keys: cthyb.update({key: self[key]})
        if cthyb['verbosity'] == 1: cthyb['verbosity'] += 1
        return cthyb

    @staticmethod
    def _missing_dmft_parameters(parameters):
        """gets dict, returns list of the missing"""
        mispar = list()
        if 'scheme' in parameters.keys():
            if 'pcdmft' in parameters['scheme']:
                CleanLoopParameters._obligatory.append('periodization')
        for par in CleanLoopParameters._obligatory:
            assert par in parameters.keys(), 'obligatory parameter \'' + par + '\' is missing'
        for par in CleanLoopParameters._replenishing_parameters.keys():
            if not(par in parameters.keys()):
                mispar.append(par)
        return mispar

    @staticmethod
    def _replenish(missing_keys):
        """gets list with missing keys and returns replenished dict"""
        pars = dict()
        for par in missing_keys:
            pars.update({par: CleanLoopParameters._replenishing_parameters[par]})
        return pars

    @staticmethod
    def _adjust(pars):
        """corrects data types for certain parameterss or adds trivial data"""
        if not tuple([0] * get_dim(pars)) in pars['t'].keys():
            pars['t'].update({tuple([0] * get_dim(pars)) : zeros([get_n_sites(pars)] * 2)})
        for key, val in pars['t'].items():
            if type(val) != ndarray:
                pars['t'][key] = array(val)
        if not pars['transformation']:
            pars['transformation'] = identity(get_n_sites(pars))
        pars['n_sites'] = get_n_sites(pars)
        pars['sites'] = range(get_n_sites(pars))
        return pars
