from matplotlib import pyplot as plt, cm
from matplotlib.backends.backend_pdf import PdfPages
from numpy import ndarray, identity, array, zeros
import os
from pytriqs.utility.dichotomy import dichotomy, mpi
from pytriqs.gf.local import BlockGf, inverse, GfReFreq, GfImFreq, GfImTime, LegendreToMatsubara, TailGf
from pytriqs.version import version
from pytriqs.archive import HDFArchive
from pytriqs.plot.mpl_interface import oplot
from pytriqs.applications.impurity_solvers.cthyb import SolverCore as Solver
from pytriqs.random_generator import random_generator_names_list
from time import time

from .archive import dict_to_archive, archive_to_dict, archive_content, load_sym_indices
from .schemes import Cellular_DMFT, PCDMFT_Li, MPCDMFT, PCDMFT_Kot
from .lattice.superlatticetools import dispersion as energy_dispersion
from .periodization import Periodization
from .plot import plot_from_archive, plot_of_loops_from_archive, checksym_plot, checktransf_plot
from .post_process_g import clip_g, tail_start, impose_site_symmetries, impose_paramagnetism, MixUpdate
from .spectrum import get_spectrum
from .transformation.gf import g_sym, g_c, sym_indices
from .transformation.operators import h_loc_sym
from .transformation.other import hop_loc_sym
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
                            p[str(key)] = archive_to_dict(p['archive'], ['parameters', str(key)])
                        else:
                            p[str(key)] = a_p[key]
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
                a_p[str(key)] = val
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
        if not tuple([0] * lattice_dim) in p['hop'].keys(): p['hop'].update({tuple([0] * lattice_dim) : zeros([lattice_dim, lattice_dim])})

        # Initialize DMFT objects
        if self.parameters['scheme'] == 'cellular_dmft':
            scheme = Cellular_DMFT(p['lattice_vectors'], p['clustersite_pos'], p['hop'], p['n_kpts'])
        elif self.parameters['scheme'] == 'pcdmft_li':
            assert 'superlattice_periodization' in self.parameters, 'superlattice_periodization required for PCDMFT_Li'
            scheme = PCDMFT_Li(p['lattice_vectors'], p['clustersite_pos'], p['hop'], p['n_kpts'], p['superlattice_periodization'])
        elif self.parameters['scheme'] == 'mpcdmft':
            assert 'superlattice_periodization' in self.parameters, 'superlattice_periodization required for PCDMFT_Li'
            scheme = MPCDMFT(p['lattice_vectors'], p['clustersite_pos'], p['hop'], p['n_kpts'], p['superlattice_periodization'])
        elif self.parameters['scheme'] == 'pcdmft_kot': # TODO written for full translation symmetry only
            assert 'hop_sublat' in self.parameters, 'hop_sublat required for PCDMFT_Kot'
            assert 'clustersite_pos_direct' in self.parameters, 'clustersite_pos_direct required for PCDMFT_Kot'
            scheme = PCDMFT_Kot(p['lattice_vectors'], [[0, 0, 0]], p['hop_sublat'], p['n_kpts'], p['clustersite_pos_direct'])
        g_c_iw = BlockGf(name_block_generator = [(s, GfImFreq(indices = sites, beta = p['beta'], n_points = p['n_iw'])) for s in CDmft._spins], name = '$G_c$')
        sigma_c_iw = BlockGf(name_block_generator = [(s, GfImFreq(indices = sites, beta = p['beta'], n_points = p['n_iw'])) for s in CDmft._spins], name = '$\Sigma_c$')
        if 'Sigma_c_iw' in p.keys(): sigma_c_iw << p['Sigma_c_iw']
        if 'mu' in p.keys():
            mu0 = p['mu']
        elif self.next_loop() > 0:
            mu0 = self.load('mu')
        else:
            mu0 = p['u'] * .5
        dmu = 0
        if self.next_loop() > 0:
            mix = MixUpdate(self.load('G_c_iw'))
        else:
            mix = MixUpdate(g_c_iw)

        # Checkout G_loc for blocks after symmetry transformation and initialize solver
        if not ('symmetry_transformation' in p.keys()): p['symmetry_transformation'] = identity(n_sites)
        sym_ind = sym_indices(scheme.g_local(sigma_c_iw, dmu), p['symmetry_transformation'])
        imp_sol = Solver(beta = p['beta'], gf_struct = dict(sym_ind), n_tau = p['n_tau'], n_iw = p['n_iw'], n_l = p['n_legendre'])
        if p['verbosity'] > 0: mpi.report('Indices for calculation are:', sym_ind)
        delta_sym_tau = imp_sol.Delta_tau.copy()
        delta_sym_tau.name = '$\Delta_{sym}$'
        g_sym_iw = BlockGf(name_block_generator = [(ind, GfImFreq(indices = dict(sym_ind)[ind], mesh = g_c_iw[CDmft._spins[0]].mesh)) for ind in dict(sym_ind).keys()], make_copies = False)
        g_0_iw = BlockGf(name_block_generator = [(ind, GfImFreq(indices = dict(sym_ind)[ind], mesh = g_c_iw[CDmft._spins[0]].mesh)) for ind in dict(sym_ind).keys()], make_copies = False)
        sigma_sym_iw = BlockGf(name_block_generator = [(ind, GfImFreq(indices = dict(sym_ind)[ind], mesh = sigma_c_iw[CDmft._spins[0]].mesh)) for ind in dict(sym_ind).keys()], make_copies = False)

        # DMFT loop
        for loop_nr in range(self.next_loop(), self.next_loop() + n_dmft_loops):       
            if mpi.is_master_node(): mpi.report('DMFT-loop nr. %s'%loop_nr)

            # Estimate mu for a given filling or vice versa
            dens = lambda dmu : scheme.g_local(sigma_c_iw, dmu).total_density()
            if 'density' in p.keys():
                if p['density']:
                    dmu, density0 = dichotomy(function = dens, x_init = dmu, 
                                             y_value = p['density'], 
                                             precision_on_y = 0.001, delta_x = 0.5,
                                             max_loops = 1000, x_name = 'dmu', 
                                             y_name = 'density', verbosity = 0)
            if mpi.is_master_node() and p['verbosity'] > 0: mpi.report('mu: %s'%(mu0 + dmu))

            # Inverse FT
            g_c_iw << scheme.g_local(sigma_c_iw, dmu)
            if mpi.is_master_node() and p['verbosity'] > 1: checksym_plot(g_c_iw, p['archive'][0:-3] + 'Gchecksym' + str(loop_nr) + '.png')

            # Use transformation to (block-)diagonalize G: G_sym = U.G.U_dag
            g_sym_iw << g_sym(g_c_iw, p['symmetry_transformation'], sym_ind)
            sigma_sym_iw << g_sym(sigma_c_iw, p['symmetry_transformation'], sym_ind)
            if mpi.is_master_node() and p['verbosity'] > 1: checktransf_plot(g_sym_iw, p['archive'][0:-3] + 'Gchecktransf' + str(loop_nr) + '.png')

            # Dyson equation for the Weiss-field
            for s, b in g_0_iw: b << inverse(sigma_sym_iw[s] + inverse(g_sym_iw[s]) + hop_loc_sym(p['hop'][(0, 0)], p['symmetry_transformation'], sym_ind)[s])
            if mpi.is_master_node() and p['verbosity'] > 1: 
                checktransf_plot(g_0_iw, p['archive'][0:-3] + 'Gweisscheck' + str(loop_nr) + '.png')
                checksym_plot(inverse(g_0_iw), p['archive'][0:-3] + 'invGweisscheckconst' + str(loop_nr) + '.png')

            # Solve imuprity problem
            rnames = random_generator_names_list()
            rname = rnames[int((loop_nr + mpi.rank) % len(rnames))]
            seed = 862379 * mpi.rank# + int(time()*10**6) - int(time()) * 10**6
            solver_parameters = ['n_cycles', 'length_cycle', 'n_warmup_cycles', 'random_seed', 'random_name', 'max_time', 'verbosity', 'use_trace_estimator', 'measure_g_tau', 'measure_g_l', 'measure_pert_order', 'make_histograms']
            pp = {'random_seed' : seed, 'random_name' : rname}
            for i in solver_parameters:
                if i in p.keys():
                    pp[i] = p[i]
            imp_sol.G0_iw << g_0_iw
            h_loc = h_loc_sym(p['u'], mu0, p['hop'][(0, 0)], p['symmetry_transformation'], sym_ind, p['verbosity'])
            imp_sol.solve(h_loc = h_loc, **pp)
            delta_sym_tau << imp_sol.Delta_tau

            # Post-processing the solver-result
            if p['measure_g_l']:
                for name, g_l in imp_sol.G_l:
                    g_sym_iw[name] << LegendreToMatsubara(g_l)
                g_sym_iw_raw = g_sym_iw.copy()
                sigma_sym_iw_raw = g_sym_iw.copy()
                for s, b in sigma_sym_iw_raw: b << inverse(g_0_iw[s]) - inverse(g_sym_iw[s]) - hop_loc_sym(p['hop'][(0, 0)], p['symmetry_transformation'], sym_ind)[s]
            else:
                for name, g_tau in imp_sol.G_tau:
                    g_sym_iw[name].set_from_fourier(g_tau)
                g_sym_iw_raw = g_sym_iw.copy()
                sigma_sym_iw_raw = g_sym_iw.copy()
                sigma_sym_iw_raw << inverse(g_0_iw) - inverse(g_sym_iw)
                if 'fit_tail' in p:
                    if p['fit_tail']:
                        for name, g in g_sym_iw:
                            for ind in sym_ind:
                                if ind[0] == name: block_inds = ind[1]
                            fixed_moments = TailGf(len(block_inds), len(block_inds), 1, 1)
                            fixed_moments[1] = identity(len(block_inds))
                            g_sym_iw[name].fit_tail(fixed_moments, 8, p['tail_start'], p['n_iw'] - 1)

            g_sym_iw << clip_g(g_sym_iw, p['clipping_threshold'])
            g_c_iw << g_c(g_sym_iw, p['symmetry_transformation'], sym_ind)
            g_c_iw << clip_g(g_c_iw, p['clipping_threshold'])
            if 'impose_paramagnetism' in p.keys(): 
                if p['impose_paramagnetism']: g_c_iw << impose_paramagnetism(g_c_iw)
            if 'site_symmetries' in p.keys(): g_c_iw << impose_site_symmetries(g_c_iw, p['site_symmetries'])
            if 'mix_coeff' in p.keys(): g_c_iw << mix(g_c_iw, p['mix_coeff'])
            g_c_iw << clip_g(g_c_iw, p['clipping_threshold'])
            g_sym_iw << g_sym(g_c_iw, p['symmetry_transformation'], sym_ind)
            for s, b in sigma_sym_iw: b << inverse(g_0_iw[s]) - inverse(g_sym_iw[s]) - hop_loc_sym(p['hop'][(0, 0)], p['symmetry_transformation'], sym_ind)[s]

            # Backtransformation to site-basis
            g_c_iw << g_c(g_sym_iw, p['symmetry_transformation'], sym_ind)
            sigma_c_iw << g_c(sigma_sym_iw, p['symmetry_transformation'], sym_ind)
            sigma_c_iw_raw = sigma_c_iw.copy()
            sigma_c_iw_raw << g_c(sigma_sym_iw_raw, p['symmetry_transformation'], sym_ind)

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
                a_l['G_sym_iw_raw'] = g_sym_iw_raw
                a_l['Sigma_c_iw'] = sigma_c_iw
                a_l['Sigma_c_iw_raw'] = sigma_c_iw_raw
                a_l['mu'] = dmu + mu0
                a_l['dmu'] = dmu
                a_l['density'] = density
                a_l['sign'] = imp_sol.average_sign
                a_l['spectrum'] = get_spectrum(imp_sol)
                a_l['eps'] = scheme.eps_rbz
                a_l['rbz_grid'] = scheme.rbz_grid
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
                a['parameters']['dmu'] = dmu
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
        prange = (0, 100)

        if 'name' in p.keys():
            filename = p['name'] + '.pdf'
        else:
            filename = p['archive'][0:-3] + '.pdf'
        pp = PdfPages(filename)

        functions = ['G_c_iw', 'Sigma_c_iw']
        for f in functions:
            for i in sites:
                plot_from_archive(p['archive'], f, [-1], indices = [(0, i)], x_window = prange, marker = '+')
                pp.savefig()
                plt.close()

        markers = ['o', '+', 'x', '^', '>', 'v', '<', '.', 'd', 'h']
        for i in sites:
            m = markers[i % len(markers)]
            plot_from_archive(p['archive'], 'G_c_iw', [-1], indices = [(i, i)], x_window = prange, marker = m)
        pp.savefig()
        plt.close()

        for m in ['R', 'I']:
            plot_from_archive(p['archive'], 'G_c_iw', range(-min(self.next_loop(), 3), 0), spins = [CDmft._spins[0]], RI = m, x_window = prange, marker = 'x')
            plot_from_archive(p['archive'], 'G_c_iw', range(-min(self.next_loop(), 3), 0), spins = [CDmft._spins[1]], RI = m, x_window = prange, marker = 'o')
            pp.savefig()
            plt.close()

        a = HDFArchive(p['archive'], 'r')
        if a['Results'][str(self.last_loop())].is_group('G_sym_l'):
            oplot(self.load('G_sym_l'))
            plt.gca().set_ylabel('$G_{sym}(l)$')
            pp.savefig()
            plt.close()
        del a

        plot_from_archive(p['archive'], 'Delta_sym_tau', range(-min(self.next_loop(), 5), 0), spins = ['0-up'])
        pp.savefig()
        plt.close()

        plot_from_archive(p['archive'], 'Delta_sym_tau', spins = load_sym_indices(p['archive'], -1))
        pp.savefig()
        plt.close()

        inds = load_sym_indices(p['archive'], -1)

        a = HDFArchive(self.parameters['archive'], 'r')
        n_graphs = 0
        for spin, orb_list in inds.items():
            n_graphs += len(orb_list)
        for m in ['R', 'I']:
            c = 0
            for ind in inds:
                for orb in inds[ind]:
                    if 'Sigma_c_iw_raw' in a['Results'][str(self.last_loop())]: plot_from_archive(p['archive'], 'G_sym_iw_raw', indices = [(orb, orb)], spins = [str(ind)], RI = m, x_window = prange, marker = 'x', color = cm.jet(c/float(n_graphs)))
                    plot_from_archive(p['archive'], 'G_sym_iw', indices = [(orb, orb)], spins = [str(ind)], RI = m, x_window = prange, marker = '+', color = cm.jet(c/float(n_graphs)))
                    c += 1
            pp.savefig()
            plt.close()
        del a

        a = HDFArchive(self.parameters['archive'], 'r')
        in_archive = False
        if 'Sigma_c_iw_raw' in a['Results'][str(self.last_loop())]:
            in_archive = True
        del a
        if in_archive:
            for m in ['R', 'I']:
                c = 0
                for i in range(n_sites):
                    for j in range(n_sites):
                        plot_from_archive(p['archive'], 'Sigma_c_iw_raw', indices = [(i, j)], spins = ['up'], RI = m, x_window = prange, marker = 'x', color = cm.jet(c/float(n_sites**2)))
                        plot_from_archive(p['archive'], 'Sigma_c_iw_raw', indices = [(i, j)], spins = ['down'], RI = m, x_window = prange, marker = '+', color = cm.jet(c/float(n_sites**2)))
                        c += 1
                pp.savefig()
                plt.close()

        for m in ['R', 'I']:
            plot_from_archive(p['archive'], 'G_c_iw', range(-min(self.next_loop(), 5), 0), RI = m, x_window = prange, marker = '+')
            pp.savefig()
            plt.close()

        for m in ['R', 'I']:
            plot_from_archive(p['archive'], 'Sigma_c_iw', range(-min(self.next_loop(), 5), 0), RI = m, x_window = prange, marker = '+')
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
            oplot(lat.get_g_lat_loc()['up'], '-+', x_window = prange)
            pp.savefig()
            plt.close()
            oplot(lat.get_sigma_lat_loc()['up'], '-+', x_window = prange)
            pp.savefig()
            plt.close()
            lat.plot_dos_loc()
            pp.savefig()
            plt.close()
            lat.plot('G_lat', (0, 0), 'up', (0, 0), '-+', x_window = prange)
            lat.plot('G_lat', (.5, 0), 'up', (0, 0), '-+', x_window = prange)
            lat.plot('G_lat', (.25, .25), 'up', (0, 0), '-+', x_window = prange)
            pp.savefig()
            plt.close()
            lat.plot('Sigma_lat', (0, 0),'up',(0, 0), '-+', x_window = prange)
            lat.plot('Sigma_lat', (.5, 0), 'up', (0, 0), '-+', x_window = prange)
            lat.plot('Sigma_lat', (.25, .25), 'up', (0, 0), '-+', x_window = prange)
            pp.savefig()
            plt.close()
            path = [[0, 0], [.5, 0], [.5, .5], [0, 0]]
            lat.plot_dos_k_w(path)
            pp.savefig()
            plt.close()
            lat.plot2d_k('Tr_G_lat', 0, imaginary_part = True)
            pp.savefig()
            plt.close()
            lat.plot2d_k('Sigma_lat', 0, 'up', 0, imaginary_part = True)
            pp.savefig()
            plt.close()
            lat.plot_hist2d_dos_k()
            pp.savefig()
            plt.close()
            lat.plot_hist2d_energy_dispersion()
            pp.savefig()
            plt.close()
        del a

        arch_text = archive_content(self.parameters['archive'], dont_exp = ['parameters', 'bz_grid', 'bz_weights', 'eps', 'rbz_grid'])
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
