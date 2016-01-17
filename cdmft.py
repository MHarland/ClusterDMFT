from itertools import izip
from matplotlib import pyplot as plt, cm
from matplotlib.backends.backend_pdf import PdfPages
from numpy import ndarray, identity, array, zeros
from pytriqs.utility import mpi
from pytriqs.gf.local import BlockGf, inverse, GfReFreq, GfImFreq, GfImTime, LegendreToMatsubara, TailGf, iOmega_n
from pytriqs.version import version
from pytriqs.archive import HDFArchive
from pytriqs.plot.mpl_interface import oplot
from pytriqs.applications.impurity_solvers.cthyb import SolverCore as Solver
from pytriqs.random_generator import random_generator_names_list
from time import time

from .archive import ArchiveConnected
from .dmft_struct import DMFTObjects
from .schemes import get_scheme
from .lattice.superlatticetools import dispersion as energy_dispersion #remove somehow
from .periodization.periodization import ClusterPeriodization
from .plot import plot_from_archive, plot_of_loops_from_archive, checksym_plot, checktransf_plot, plot_ln_abs #move to dmftobjects?
from .loop_parameters import CleanLoopParameters
from .process_g import addExtField
from .transformation.sites import ClustersiteTransformation
from .transformation.nambuplaquette import NambuPlaquetteTransformation
from .utility import get_site_nrs, get_dim, get_n_sites, Reporter

class CDmft(ArchiveConnected):
    """
    TODO
    Transformation has to be unitary(?)
    """

    _version = '1.10'

    def __init__(self, **kwargs):
        self.parameters = dict()
        for i in kwargs:
            self.parameters[i] = kwargs[i]
        super(CDmft, self).__init__(**self.parameters)
        if mpi.is_master_node():
            archive = HDFArchive(self.parameters['archive'], 'a')
            archive['CDmft_version'] = CDmft._version
            del archive

    def set_parameters(self, parameters):
        """parameters: dict"""
        for key, val in parameters.items():
            self.parameters[key] = val

    def get_parameter(self, key):
        return self.parameters[key]

    def get_parameters(self):
        return self.parameters

    def print_parameters(self):
        if mpi.is_master_node():
            print 'Parameters:'
            print self.parameters
            print

    def run_dmft_loops(self, n_dmft_loops = 1):
        """runs the DMFT calculation"""
        clp = p = CleanLoopParameters(self.get_parameters())
        report = Reporter(**clp)
        report('Parameters:', clp)
        scheme = get_scheme(clp)
        dmft = DMFTObjects(**clp)
        raw_dmft = DMFTObjects(**clp)
        g_c_iw, sigma_c_iw, g_0_c_iw, dmu = dmft.get_dmft_objs()

        report('Initializing...')
        if 'nambu' in clp['scheme']:
            transf = NambuPlaquetteTransformation(**clp)
            raw_transf = NambuPlaquetteTransformation(**clp)
            g_0_c_iw << 0
        else:
            transf = ClustersiteTransformation(g_loc = scheme.g_local(sigma_c_iw, dmu), **clp)
            clp.update({'g_transf_struct': transf.get_g_struct()})
            raw_transf = ClustersiteTransformation(**clp)
        transf.set_hamiltonian(**clp)
        report(transf.hamiltonian)
        report('Transformation ready')
        report('New basis:', transf.get_g_struct())
        impurity = Solver(beta = clp['beta'], gf_struct = dict(transf.get_g_struct()), 
                          n_tau = clp['n_tau'], n_iw = clp['n_iw'], n_l = clp['n_legendre'])
        impurity.Delta_tau.name = '$\\tilde{\\Delta}_c$'
        rnames = random_generator_names_list()
        report('Impurity solver ready')

        for loop_nr in range(self.next_loop(), self.next_loop() + n_dmft_loops):       
            report('DMFT-loop nr. %s'%loop_nr)
            if mpi.is_master_node(): duration = time()

            dmft.find_dmu(scheme, **clp)
            report('dmu: %s'%dmu)
            g_c_iw, sigma_c_iw, g_0_c_iw, dmu = dmft.get_dmft_objs()

            report('Calculating local Greenfunction...')
            g_c_iw << scheme.g_local(sigma_c_iw, dmu)
            g_c_iw << addExtField(g_c_iw, p['ext_field'])
            if mpi.is_master_node() and p['verbosity'] > 1: checksym_plot(g_c_iw, p['archive'][0:-3] + 'Gchecksym' + str(loop_nr) + '.pdf')
            report('Calculating Weiss-field...')
            g_0_c_iw << inverse(inverse(g_c_iw) + sigma_c_iw)
            report('Changing basis...')
            transf.set_dmft_objs(inverse(sigma_c_iw + inverse(g_c_iw)), g_c_iw, sigma_c_iw)
            """
            if mpi.is_master_node():
                ddd = impurity.Delta_tau.copy()
                ddh = transf.g_0_iw.copy()
                ddh << inverse(transf.g_0_iw) - iOmega_n
                for s,b in ddd:
                    b.set_from_inverse_fourier(ddh[s])
                for s, b in ddd:
                    for i in range(len(b.data[0,:,:])):
                        oplot(b[i,i], RI='R')
                plt.gca().set_ylim(-5,2)
                plt.savefig('delt'+str(loop_nr)+'.pdf')
                plt.close()
            """

            if mpi.is_master_node() and p['verbosity'] > 1: 
                checktransf_plot(transf.get_g_iw(), p['archive'][0:-3] + 'Gchecktransf' + str(loop_nr) + '.pdf')
                checktransf_plot(g_0_c_iw, p['archive'][0:-3] + 'Gweisscheck' + str(loop_nr) + '.pdf')
                checksym_plot(inverse(transf.g_0_iw), p['archive'][0:-3] + 'invGweisscheckconst' + str(loop_nr) + '.pdf')
                checksym_plot(inverse(transf.get_g_iw()), p['archive'][0:-3] + 'invGsymcheckconst' + str(loop_nr) + '.pdf')

            if not clp['random_name']: clp.update({'random_name': rnames[int((loop_nr + mpi.rank) % len(rnames))]})
            if not clp['random_seed']: clp.update({'random_seed': 862379 * mpi.rank + 12563 * self.next_loop()})
            impurity.G0_iw << transf.get_g_0_iw()
            report('Solving impurity problem...')
            mpi.barrier()
            impurity.solve(h_int = transf.get_hamiltonian(), **clp.get_cthyb_parameters())
            """
            except:
                for s, b in impurity.Delta_tau:
                    for i in range(len(b.data[0,:,:])):
                        oplot(b[i,i], RI='R')
                plt.savefig('delta'+str(loop_nr)+'.pdf')
                plt.close()
            """
                
            if mpi.is_master_node() and p['verbosity'] > 1: 
                checksym_plot(inverse(impurity.G0_iw), p['archive'][0:-3] + 'invGweisscheckconstsolver' + str(loop_nr) + '.pdf')
            report('Postprocessing measurements...')
            if clp['measure_g_l']:
                for ind, g in transf.get_g_iw(): g  << LegendreToMatsubara(impurity.G_l[ind])
            else:
                for ind, g in transf.get_g_iw(): g.set_from_fourier(impurity.G_tau[ind])
            raw_transf.set_dmft_objs(transf.get_g_0_iw(),
                                     transf.get_g_iw(),
                                     inverse(transf.get_g_0_iw()) - inverse(transf.get_g_iw()))
            if clp['measure_g_tau'] and clp['fit_tail']:
                for ind, g in transf.get_g_iw():
                    for tind in transf.get_g_struct():
                        if tind[0] == ind: block_inds = tind[1]
                    fixed_moments = TailGf(len(block_inds), len(block_inds), 1, 1)
                    fixed_moments[1] = identity(len(block_inds))
                    g.fit_tail(fixed_moments, 3, clp['tail_start'], clp['n_iw'] - 1)
            if mpi.is_master_node() and p['verbosity'] > 1: checksym_plot(inverse(transf.get_g_iw()), p['archive'][0:-3] + 'invGsymcheckconstsolver' + str(loop_nr) + '.pdf')
            report('Backtransforming...')
            transf.set_sigma_iw(inverse(transf.get_g_0_iw()) - inverse(transf.get_g_iw()))
            dmft.set_dmft_objs(*transf.get_backtransformed_dmft_objs())
            dmft.set_dmu(dmu)
            raw_dmft.set_dmft_objs(*raw_transf.get_backtransformed_dmft_objs())
            raw_dmft.set_dmu(dmu)

            if clp['mix']: dmft.mix()
            if clp['impose_paramagnetism']: dmft.paramagnetic()
            if clp['impose_afm']: dmft.afm()
            if clp['site_symmetries']: dmft.site_symmetric(clp['site_symmetries'])
            density = dmft.get_g_iw().total_density()
            report('Saving results...')
            if mpi.is_master_node():
                a = HDFArchive(p['archive'], 'a')
                if not a.is_group('results'):
                    a.create_group('results')
                a_r = a['results']
                a_r.create_group(str(loop_nr))
                a_l = a_r[str(loop_nr)]
                a_l['delta_transf_tau'] = impurity.Delta_tau
                if clp['measure_g_l']: a_l['g_transf_l'] = impurity.G_l
                if clp['measure_g_tau']: a_l['g_transf_tau'] = impurity.G_tau
                a_l['g_c_iw'] = dmft.get_g_iw()
                a_l['g_c_iw_raw'] = raw_dmft.get_g_iw()
                a_l['g_transf_iw'] = transf.get_g_iw()
                a_l['g_transf_iw_raw'] = raw_transf.get_g_iw()
                a_l['sigma_c_iw'] = dmft.get_sigma_iw()
                a_l['sigma_c_iw_raw'] = raw_dmft.get_sigma_iw()
                a_l['sigma_transf_iw'] = transf.get_sigma_iw()
                a_l['sigma_transf_iw_raw'] = raw_transf.get_sigma_iw()
                a_l['g_0_c_iw'] = dmft.get_g_0_iw()
                a_l['g_0_c_iw_raw'] = raw_dmft.get_g_0_iw()
                a_l['g_0_transf_iw'] = transf.get_g_0_iw()
                a_l['g_0_transf_iw_raw'] = raw_transf.get_g_0_iw()
                a_l['dmu'] = dmft.get_dmu()
                a_l['density'] = density
                a_l['sign'] = impurity.average_sign
                #if clp['measure_density_matrix']: a_l['density_matrix'] = impurity.state_trace_contribs
                a_l['g_atomic_tau'] = impurity.atomic_gf
                a_l['loop_time'] = {'seconds': time() - duration,
                                    'hours': (time() - duration)/3600., 
                                    'days': (time() - duration)/3600./24.}
                a_l['n_cpu'] = mpi.size
                a_l['cdmft_code_version'] = CDmft._version
                a_l['local'] = impurity.h_loc_diagonalization
                clp_dict = dict()
                clp_dict.update(clp)
                a_l['parameters'] = clp_dict
                a_l['triqs_code_version'] = version
                if a_r.is_data('n_dmft_loops'):
                    a_r['n_dmft_loops'] += 1
                else:
                    a_r['n_dmft_loops'] = 1
                del a_l, a_r, a
            report('Loop done')
            report('')
            mpi.barrier()

    def export_results(self, filename = False, prange = (0, 60)):
        """
        routine to export easily a selection of results into a pdf file
        """        
        if mpi.is_master_node(): self._export_results(filename = filename, prange = prange)

    def _export_results(self, filename = False, prange = (0, 60)):
        p = self.parameters = self.load('parameters')
        p['archive'] = self.archive
        n_sites = len(p['blockstates'])
        sites = p['blockstates']
        blocks = p['blocks']
        transf_blocks = p['g_transf_struct']

        if not filename: filename = p['archive'][0:-3] + '.pdf'
        pp = PdfPages(filename)

        functions = ['g_c_iw', 'sigma_c_iw']
        for f in functions:
            for i in sites:
                plot_from_archive(p['archive'], f, [-1], indices = [(0, i)], x_window = prange, marker = '+', blocks = [blocks[0]])
                pp.savefig()
                plt.close()

        markers = ['o', '+', 'x', '^', '>', 'v', '<', '.', 'd', 'h']
        for i in sites:
            m = markers[i % len(markers)]
            plot_from_archive(p['archive'], 'g_c_iw', [-1], indices = [(i, i)], x_window = prange, marker = m, blocks = [blocks[0]])
        pp.savefig()
        plt.close()

        for m in ['R', 'I']:
            for b, marker in zip(blocks, ['x','o']):
                plot_from_archive(p['archive'], 'g_c_iw', range(-min(self.next_loop(), 3), 0), blocks = [blocks[0]], RI = m, x_window = prange, marker = marker)
            pp.savefig()
            plt.close()

        a = HDFArchive(p['archive'], 'r')
        if a['results'][str(self.last_loop())].is_group('g_transf_l'):
            plot_ln_abs(self.load('g_transf_l'))
            pp.savefig()
            plt.close()
        del a

        plot_from_archive(p['archive'], 'delta_transf_tau', range(-min(self.next_loop(), 5), 0), blocks = [transf_blocks[0][0]])
        pp.savefig()
        plt.close()

        plot_from_archive(p['archive'], 'delta_transf_tau', blocks = dict(p['g_transf_struct']).keys())
        pp.savefig()
        plt.close()

        inds = dict(p['g_transf_struct'])

        a = HDFArchive(self.parameters['archive'], 'r')
        n_graphs = 0
        for spin, orb_list in inds.items():
            n_graphs += len(orb_list)
        for f in ['g_transf_iw', 'sigma_transf_iw']:
            for m in ['R', 'I']:
                c = 0
                for ind in inds:
                    for orb in inds[ind]:
                        plot_from_archive(p['archive'], f+'_raw', indices = [(orb, orb)], blocks = [str(ind)], RI = m, x_window = prange, marker = 'x', color = cm.jet(c/float(n_graphs-1)))
                        plot_from_archive(p['archive'], f, indices = [(orb, orb)], blocks = [str(ind)], RI = m, x_window = prange, marker = '+', color = cm.jet(c/float(n_graphs-1)))
                        c += 1
                pp.savefig()
                plt.close()
        del a

        for f in ['g_c_iw_raw', 'sigma_c_iw_raw']:
            for m in ['R', 'I']:
                c = 0
                for i in range(n_sites):
                    for j in range(n_sites):
                        for k, b in enumerate(blocks):
                            mark = ['x', '+'][k%2]
                            plot_from_archive(p['archive'], f, indices = [(i, j)], blocks = [b], RI = m, x_window = prange, marker = mark, color = cm.jet(int(c)/float(-1+n_sites**len(blocks))))
                            c += .5
                pp.savefig()
                plt.close()

        for m in ['R', 'I']:
            plot_from_archive(p['archive'], 'g_c_iw', range(-min(self.next_loop(), 5), 0), RI = m, x_window = prange, marker = '+', blocks = [blocks[0]])
            pp.savefig()
            plt.close()

        for m in ['R', 'I']:
            plot_from_archive(p['archive'], 'sigma_c_iw', range(-min(self.next_loop(), 5), 0), RI = m, x_window = prange, marker = '+', blocks = [blocks[0]])
            pp.savefig()
            plt.close()

        functions = ['g_c_iw', 'sigma_c_iw']
        for f in functions:
            plot_of_loops_from_archive(p['archive'], f, indices = [(0, i) for i in sites], marker = '+', blocks = [blocks[0]])
            pp.savefig()
            plt.close()
        functions = ['density', 'sign', 'dmu']
        for f in functions:
            plot_of_loops_from_archive(p['archive'], f, marker = '+', blocks = [blocks[0]])
            pp.savefig()
            plt.close()
        """
        arch_text = self.archive_content(group = ['results', str(self.last_loop())], dont_exp = ['bz_grid', 'bz_weights', 'eps', 'rbz_grid'])
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
        """
        pp.close()
        print filename, 'ready'

    def plot(self, function, *args, **kwargs):
        """
        plot several loops and/or indices in one figure
        possible options for function are:
        'delta_transf_tau', 'g_c_iw', 'g_c_tau', 'sigma_c_iw', 'g_transf_l'
        further keywordarguments go into oplot
        """
        if 'filename' in kwargs.keys():
            filename = kwargs['filename']
        else:
            filename = function + '.pdf'
        plot_from_archive(self.parameters['archive'], function, *args, **kwargs)
        plt.savefig(filename)
        plt.close()
        print filename, ' ready'        

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
            filename = function + '.pdf'
        plot_of_loops_from_archive(self.parameters['archive'], function, *args, **kwargs)
        plt.savefig(filename)
        plt.close()
        print filename, ' ready'
