from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy import ndarray
import os
from pytriqs.utility.dichotomy import dichotomy, mpi
from pytriqs.gf.local import BlockGf, inverse, GfReFreq, GfImFreq, GfImTime, LegendreToMatsubara
from pytriqs.version import version
from pytriqs.archive import HDFArchive
from pytriqs.plot.mpl_interface import oplot
from pytriqs.applications.impurity_solvers.cthyb import SolverCore as Solver
from pytriqs.random_generator import random_generator_names_list
from time import time

from .archive import dict_to_archive, archive_to_dict, archive_content, load_sym_indices
from .lattice.superlatticetools import _init_k_sum, dispersion as energy_dispersion
from .periodization.cumulant_periodization import Periodization
from .plot import plot_from_archive, plot_of_loops_from_archive, checksym_plot, checktransf_plot
from .post_process_g import clip_g, tail_start
from .transformation.gf import g_sym, g_c, sym_indices
from .transformation.operators import h_loc_sym
from .transformation.u_matrix import CoulombTensor

class CDmft(object):
    """
    clustersite_pos has a lattice basis (and so has lattice_vectors)
    You can choose between density and mu. One is adjusted every dmft loop to obtain the value of the other one.
    In doubt use keywordarguments!
    Don't change the archive's filename!
    Transformation has to be unitary!
    Setting name you are garuanteed to create a new archive not overwriting any old ones. Setting archive instead let you load data from an existing archive into the CDmft instance.
    """

    _spins = ['up', 'down']
    _version = '1.00'

    def __init__(self, **kwargs):
        assert 'name' in kwargs or 'archive' in kwargs, 'CDmft instance must get a name or an archive'
        p = self.parameters = dict()
        for i in kwargs:
            p[i] = kwargs[i]
        self.sync_with_archive()

    def sync_with_archive(self):
        """
        Synchronizes the parameters that will be used for the next dmft run.
        Loads all parameters that are not defined in the instance from the archive and overwrites parameters in the archive with parameters of the instance.
        Sigma and mu will be taken from the last loop by default.
        """
        if mpi.is_master_node():
            self._sync_with_archive()
        self.parameters = mpi.bcast(self.parameters)

    def _sync_with_archive(self):
        p = self.parameters

        if 'archive' in p.keys():
            if os.path.exists(p['archive']):
                archive = HDFArchive(p['archive'], 'a')
                a_p = archive['parameters']
                for key in a_p:
                    if key not in p.keys():
                        if key == 'hop':
                            p[key] = archive_to_dict(p['archive'], ['parameters', str(key)])
                        else:
                            p[key] = a_p[key]
            else:
                archive = HDFArchive(p['archive'], 'w')
                archive.create_group('parameters')
                a_p = archive['parameters']
        else:
            aname = p['name'] + '.h5'
            nr = 1
            while os.path.exists(aname):
                aname = p['name'] + str(nr) +'.h5'
                nr += 1
            archive = HDFArchive(aname, 'w')
            archive.create_group('parameters')
            a_p = archive['parameters']
            p['archive'] = aname

        for key, val in p.items():
            if key == 'hop':
                dict_to_archive(val, p['archive'], ['parameters', str(key)])
            else:
                a_p[key] = val
        del archive

    def print_parameters(self):
        """
        prints the parameters for the next run without linebreaks
        """
        show_p = str()
        for key, val in self.parameters.items():
            show_p += str(key) + ': ' + str(val) + ', '
        if mpi.is_master_node(): mpi.report('Parameters: ' + show_p[0:-2])

    def run_dmft_loops(self, n_dmft_loops = 1):
        """
        start calculation
        """
        p = self.parameters
        if mpi.is_master_node() and p['verbosity'] > 0: self.print_parameters()
        n_sites = len(p['clustersite_pos'])
        sites = range(n_sites)
        lattice_dim = len(p['hop'].keys()[0])

        if mpi.is_master_node(): duration = time()

        # checks
        for key, val in self.parameters['hop'].items():
            assert type(val) == ndarray, 'The hoppingtensor has to be a dict with tuples as keys and arrays as values'
        if not tuple([0] * lattice_dim) in p['hop'].keys(): p['hop'].update({tuple([0] * lattice_dim) : array([[0]])})

        # Initialize DMFT objects
        k_sum = _init_k_sum(p['lattice_vectors'], p['clustersite_pos'], p['hop'], p['n_kpts'])
        if 'scheme' in self.parameters:
            if self.parameters['scheme'] == 'PCDMFT':
                clustersite_pos_cart = list()
                for dlb in p['clustersite_pos']:
                    clustersite_pos_cart.append([sum_list([p['lattice_vectors'][i][j] * dlb[j] for j in range(len(dlb))]) for i in range(len(dlb))])
                periodized = Periodization(p['sublattice_vectors'], p['sublattice_basis'], p['hop_c'], p['sublattice_n_kpts'], clustersite_pos_cart)
        g_c_iw = BlockGf(name_block_generator = [(s, GfImFreq(indices = sites, beta = p['beta'], n_points = p['n_iw'])) for s in CDmft._spins], name = '$G_c$')
        sigma_c_iw = BlockGf(name_block_generator = [(s, GfImFreq(indices = sites, beta = p['beta'], n_points = p['n_iw'])) for s in CDmft._spins], name = '$\Sigma_c$')
        sigma_old_c_iw = BlockGf(name_block_generator = [(s, GfImFreq(indices = sites, beta = p['beta'], n_points = p['n_iw'])) for s in CDmft._spins], name = '$\Sigma_old_c$')
        if 'Sigma_c_iw' in p.keys(): sigma_c_iw << p['Sigma_c_iw']
        sigma_old_c_iw << sigma_c_iw
        mu = p['mu']

        # Checkout G_loc for blocks after symmetry transformation and initialize solver
        if not ('symmetry_transformation' in p.keys()):
            p['symmetry_transformation'] = identity(n_sites)
        sym_ind = sym_indices(g_c_iw, p['symmetry_transformation'])
        imp_sol = Solver(beta = p['beta'], gf_struct = dict(sym_ind), n_tau = p['n_tau'], n_iw = p['n_iw'], n_l = p['n_legendre'])
        if p['verbosity'] > 0: mpi.report('Indices for calculation are:', sym_ind)
        delta_sym_tau = imp_sol.Delta_tau.copy()
        delta_sym_tau.name = '$\Delta_{sym}$'
        g_sym_iw = BlockGf(name_block_generator = [(ind, GfImFreq(indices = dict(sym_ind)[ind], mesh = g_c_iw[CDmft._spins[0]].mesh)) for ind in dict(sym_ind).keys()], make_copies = False)
        sigma_sym_iw = BlockGf(name_block_generator = [(ind, GfImFreq(indices = dict(sym_ind)[ind], mesh = sigma_c_iw[CDmft._spins[0]].mesh)) for ind in dict(sym_ind).keys()], make_copies = False)
        # DMFT loop
        for loop_nr in range(self.next_loop(), self.next_loop() + n_dmft_loops):       
            if mpi.is_master_node(): mpi.report('DMFT-loop nr. %s'%loop_nr)

            # Estimate mu for a given filling or vice versa
            dens = lambda mu : k_sum(mu = mu, Sigma = sigma_c_iw).total_density()
            if 'density' in p.keys():
                if p['density']:
                    mu, density0 = dichotomy(function = dens, x_init = mu, 
                                             y_value = p['density'], 
                                             precision_on_y = 0.001, delta_x = 0.5,
                                             max_loops = 1000, x_name = 'mu', 
                                             y_name = 'density', verbosity = 0)
            if mpi.is_master_node() and p['verbosity'] > 0: mpi.report('mu: %s'%mu)

            # inverse FT
            if 'scheme' in self.parameters:
                if self.parameters['scheme'] == 'PCDMFT':
                    g_c_iw << periodized.sum_k(sigma_c_iw, mu)
                else:
                    self.parameters['scheme'] = 'CellularDMFT'
                    g_c_iw << k_sum(mu = mu, Sigma = sigma_c_iw)
            else:
                self.parameters['scheme'] = 'CellularDMFT'
                g_c_iw << k_sum(mu = mu, Sigma = sigma_c_iw)
            if mpi.is_master_node() and p['verbosity'] > 1: checksym_plot(g_c_iw, p['archive'][0:-3] + 'Gchecksym' + str(loop_nr) + '.png')

            # Use transformation to (block-)diagonalize G: G_sym = U.G.U_dag
            g_sym_iw << g_sym(g_c_iw, p['symmetry_transformation'], sym_ind)
            sigma_sym_iw << g_sym(sigma_c_iw, p['symmetry_transformation'], sym_ind)
            if mpi.is_master_node() and p['verbosity'] > 1: checktransf_plot(g_sym_iw, p['archive'][0:-3] + 'Gchecktransf' + str(loop_nr) + '.png')

            # Dyson equation for the Weiss-field
            _temp = inverse(sigma_sym_iw + inverse(g_sym_iw))
            for key, block in _temp:
                imp_sol.G0_iw[key] << block
            del _temp
            if mpi.is_master_node() and p['verbosity'] > 1: checktransf_plot(imp_sol.G0_iw, p['archive'][0:-3] + 'Gweisscheck' + str(loop_nr) + '.png')

            # Solve imuprity problem
            rnames = random_generator_names_list()
            rname = rnames[int((loop_nr + mpi.rank) % len(rnames))]
            seed = int(time()*10**6) - int(time()) * 10**6 + 862379 * mpi.rank
            solver_parameters = ['n_cycles', 'length_cycle', 'n_warmup_cycles', 'random_seed', 'random_name', 'max_time', 'verbosity', 'use_trace_estimator', 'measure_g_tau', 'measure_g_l', 'measure_pert_order', 'make_histograms']
            pp = {'random_seed' : seed, 'random_name' : rname}
            for i in solver_parameters:
                if i in p.keys():
                    pp[i] = p[i]
            h_loc = h_loc_sym(p['u'], p['hop'][(0, 0)], p['symmetry_transformation'], sym_ind, p['verbosity'])
            imp_sol.solve(h_loc = h_loc, **pp)
            delta_sym_tau << imp_sol.Delta_tau

            # transform  and tail-fit, clipping almost zero to exact zero in both basis, Dyson eq. for the self-energy
            if p['measure_g_l']:
                for name, g_l in imp_sol.G_l:
                    g_sym_iw[name] << LegendreToMatsubara(g_l)

                    g_sym_iw_unfitted = g_sym_iw.copy()
                    if 'fit_tail' in p:
                        if p['fit_tail']:
                            for ind in sym_ind:
                                if ind[0] == name: block_inds = ind[1]
                            fixed_moments = TailGf(len(block_inds), len(block_inds), 1, 1)
                            fixed_moments[1] = identity(len(block_inds))
                            g_sym_iw[name].fit_tail(fixed_moments, 8, tail_start(g_sym_iw[name], 10), p['n_iw'] - 1)

                for name, g in g_sym_iw:
                    _temp = g.copy()
                    g_sym_iw[name] = clip_g(_temp, p['clipping_threshold'])
                    del _temp

                g_c_iw << g_c(g_sym_iw, p['symmetry_transformation'], sym_ind)
                for name, g in g_c_iw:
                    _temp = g.copy()
                    g_c_iw[name] = clip_g(_temp, p['clipping_threshold'])
                    del _temp
                g_sym_iw << g_sym(g_c_iw, p['symmetry_transformation'], sym_ind)

                for name, block in g_sym_iw:
                    sigma_sym_iw[name] << inverse(imp_sol.G0_iw[name]) - inverse(g_sym_iw[name])

            else:
                for n, b in imp_sol.G_iw:
                    g_sym_iw[n] << b
                for n, b in imp_sol.Sigma_iw:
                    sigma_sym_iw[n] << b

            # Backtransformation to site-basis
            g_c_iw << g_c(g_sym_iw, p['symmetry_transformation'], sym_ind)
            sigma_c_iw << g_c(sigma_sym_iw, p['symmetry_transformation'], sym_ind)

            # Impose paramagnetism
            if 'impose_paramagnetism' in p.keys():
                if p['impose_paramagnetism']:
                    _temp = sigma_c_iw.copy()
                    for s in CDmft._spins:
                        sigma_c_iw[s] << .5 * (_temp[CDmft._spins[0]] + _temp[CDmft._spins[1]])

            # mix self-energy
            _temp = sigma_c_iw.copy()
            for key, block in _temp:
                sigma_c_iw[key] << p['mix_coeff'] * block + (1 - p['mix_coeff']) * sigma_old_c_iw[key]
                sigma_old_c_iw[key] << sigma_c_iw[key]

            density = g_c_iw.total_density()
            if mpi.is_master_node():
                mpi.report('Density per cluster: ' + str(density))

            # saves
            if mpi.is_master_node():
                a = HDFArchive(p['archive'], 'a')
                if not a.is_group('Results'):
                    a.create_group('Results')
                a_r = a['Results']
                a_r.create_group(str(loop_nr))
                a_l = a_r[str(loop_nr)]
                a_l['Delta_sym_tau'] = delta_sym_tau
                if 'measure_g_l' in p.keys(): 
                    if p['measure_g_l']: 
                        a_l['G_sym_l'] = imp_sol.G_l
                a_l['G_c_iw'] = g_c_iw
                a_l['G_sym_iw'] = g_sym_iw
                a_l['G_sym_iw_unfitted'] = g_sym_iw_unfitted
                a_l['Sigma_c_iw'] = sigma_c_iw
                a_l['mu'] = mu
                a_l['density'] = density
                a_l['sign'] = imp_sol.average_sign
                a_l.create_group('sym_indices')
                a_l['sym_indices'].update(dict(sym_ind))
                duration = time() - duration
                a_l['loop_time'] = str(duration)
                a_l['n_cpu'] = mpi.size
                a_l['cdmft_code_version'] = CDmft._version
                a_l['parameters'] = self.parameters
                a_l['triqs_code_version'] = version
                if a_r.is_data('n_dmft_loops'):
                    a_r['n_dmft_loops'] += 1
                else:
                    a_r['n_dmft_loops'] = 1
                a['parameters']['Sigma_c_iw'] = sigma_c_iw
                a['parameters']['mu'] = mu
                del a
                mpi.report('')

    def export_results(self):
        """
        routine to export easily a selection of results into a pdf file
        """
        if mpi.is_master_node(): self._export_results()

    def _export_results(self):
        p = self.parameters
        n_sites = len(p['clustersite_pos'])
        sites = range(n_sites)

        if 'name' in p.keys():
            filename = p['name'] + '.pdf'
        else:
            filename = p['archive'][0:-3] + '.pdf'
        pp = PdfPages(filename)

        functions = ['G_c_iw', 'Sigma_c_iw']
        for f in functions:
            for i in sites:
                plot_from_archive(p['archive'], f, [-1], indices = [(0, i)], x_window = (0, 40), marker = '+')
                pp.savefig()
                plt.close()

        markers = ['o', '+', 'x', '^', '>', 'v', '<']
        for i in sites:
            m = markers[i % len(markers)]
            plot_from_archive(p['archive'], 'G_c_iw', [-1], indices = [(i, i)], x_window = (0, 40), marker = m)
        pp.savefig()
        plt.close()

        for m in ['R', 'I']:
            plot_from_archive(p['archive'], 'G_c_iw', range(-min(self.next_loop(), 3), 0), spins = [CDmft._spins[0]], RI = m, x_window = (0, 40), marker = 'x')
            plot_from_archive(p['archive'], 'G_c_iw', range(-min(self.next_loop(), 3), 0), spins = [CDmft._spins[1]], RI = m, x_window = (0, 40), marker = 'o')
            pp.savefig()
            plt.close()

        a = HDFArchive(p['archive'], 'r')
        if a['Results'][str(self.last_loop())].is_group('G_sym_l'):
            plot_from_archive(p['archive'], 'G_sym_l', spins = dict(load_sym_indices(p['archive'], -1)).keys())
            pp.savefig()
            plt.close()
        del a

        plot_from_archive(p['archive'], 'Delta_sym_tau', range(-min(self.next_loop(), 5), 0), spins = ['0-up'])
        pp.savefig()
        plt.close()

        plot_from_archive(p['archive'], 'Delta_sym_tau', spins = load_sym_indices(p['archive'], -1))
        pp.savefig()
        plt.close()

        for m in ['R', 'I']:
            plot_from_archive(p['archive'], 'G_sym_iw_unfitted', spins = load_sym_indices(p['archive'], -1), RI = m, x_window = (0, 40), marker = '+')
            plot_from_archive(p['archive'], 'G_sym_iw', spins = load_sym_indices(p['archive'], -1), RI = m, x_window = (0, 40), marker = '+')
            pp.savefig()
            plt.close()

        for m in ['R', 'I']:
            plot_from_archive(p['archive'], 'G_c_iw', range(-min(self.next_loop(), 5), 0), RI = m, x_window = (0, 40), marker = '+')
            pp.savefig()
            plt.close()

        functions = ['G_c_iw', 'Sigma_c_iw', 'mu', 'density']
        a = HDFArchive(self.parameters['archive'], 'r')
        if 'sign' in a['Results'][str(self.last_loop())]: functions.append('sign')
        del a
        for f in functions:
            plot_of_loops_from_archive(p['archive'], f, marker = '+')
            pp.savefig()
            plt.close()

        a = HDFArchive(self.parameters['archive'], 'r')
        if a.is_group('Periodization'):
            lat = Periodization(archive = self.parameters['archive'])
            lat.plot_dos_loc()
            pp.savefig()
            plt.close()
            oplot(lat.get_g_lat_loc()['up'], '-+', x_window = (0, 40))
            pp.savefig()
            plt.close()
            oplot(lat.get_sigma_lat_loc()['up'], '-+', x_window = (0, 40))
            pp.savefig()
            plt.close()
            lat.plot('Sigma_lat', (0, 0),'up',(0, 0), '-+', x_window = (0, 40))
            lat.plot('Sigma_lat', (.5, 0), 'up', (0, 0), '-+', x_window = (0, 40))
            lat.plot('Sigma_lat', (.25, .25), 'up', (0, 0), '-+', x_window = (0, 40))
            pp.savefig()
            plt.close()
            lat.plot('G_lat', (0, 0), 'up', (0, 0), '-+', x_window = (0, 40))
            lat.plot('G_lat', (.5, 0), 'up', (0, 0), '-+', x_window = (0, 40))
            lat.plot('G_lat', (.25, .25), 'up', (0, 0), '-+', x_window = (0, 40))
            pp.savefig()
            plt.close()
            lat.plot_hist2d_dos_k()
            pp.savefig()
            plt.close()
            lat.plot2d_k('Tr_G_lat', 0, imaginary_part = True)
            pp.savefig()
            plt.close()
            lat.plot2d_k('Sigma_lat', 0, 'up', 0, imaginary_part = True)
            pp.savefig()
            plt.close()
            path = [[0, 0], [.5, 0], [.5, .5], [0, 0]]
            lat.plot_dos_k_w(path)
            pp.savefig()
            plt.close()
            lat.plot_hist2d_energy_dispersion()
            pp.savefig()
            plt.close()
        del a

        arch_text = archive_content(self.parameters['archive'], dont_exp = ['parameters', 'bz_grid', 'bz_weights'])
        arch_text += archive_content(self.parameters['archive'], group = ['parameters'])
        line = 1
        page_text = str()
        for nr, char in enumerate(arch_text, 2):
            page_text += char
            if '\n' in arch_text[(nr - 2):nr]:
                line += 1
            if line % 80 == 0 or nr == len(arch_text) - 1:
                plt.figtext(0.05, 0.95, page_text, fontsize = 8, verticalalignment = 'top')
                pp.savefig()
                plt.close()
                page_text = str()
                line = 1

        pp.close()

    def plot(self, function, *args, **kwargs):
        """
        plot several loops and/or indices in one figure
        possible options for function are:
        'Delta_sym_tau', 'G_c_iw', 'G_c_tau', 'Sigma_c_iw', 'G_sym_l'
        further keywordarguments go into oplot
        """
        if 'filename' in kwargs.keys():
            filename = kwargs['filename']
        else:
            filename = function + '.png'
        plot_from_archive(self.parameters['archive'], function, *args, **kwargs)
        plt.savefig(filename)
        plt.close()
        print filename, 'ready'        

    def plot_of_loops(self, function, *args, **kwargs):
        """
        Options for function are: all functions of iw, mu and density.
        matsubara_freqs: list of int
        blocks: list of str
        indices: list of 2-tuple
        RI: 'R', 'I'
        further keywordarguments go to plot of matplotlib.pyplot
        """
        if 'filename' in kwargs.keys():
            filename = kwargs['filename']
        else:
            filename = function + '.png'
        plot_of_loops_from_archive(self.parameters['archive'], function, *args, **kwargs)
        plt.savefig(filename)
        plt.close()
        print filename, 'ready'        

    def next_loop(self):
        a = HDFArchive(self.parameters['archive'], 'r')
        if a.is_group('Results'):
            nl = a['Results']['n_dmft_loops']
        else:
            nl = 0
        del a
        return nl

    def last_loop(self):
        a = HDFArchive(self.parameters['archive'], 'r')
        ll = a['Results']['n_dmft_loops'] - 1
        del a
        return ll

    def load(self, function_name, loop_nr = -1):
        """
        returns a calculated function from archive
        function_name: 'Sigma_c_iw', 'G_c_iw', ...
        loop_nr: int
        """
        a = HDFArchive(self.parameters['archive'], 'r')
        if loop_nr < 0:
            function = a['Results'][str(self.next_loop() + loop_nr)][function_name]
        else:
            function = a['Results'][str(loop_nr)][function_name]
        del a
        return function
