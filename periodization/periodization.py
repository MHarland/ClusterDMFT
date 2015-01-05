from pytriqs.gf.local import BlockGf, GfImFreq, iOmega_n, inverse, GfReFreq
from pytriqs.gf.local.descriptor_base import Const
from pytriqs.archive import HDFArchive
from pytriqs.plot.mpl_interface import oplot
import pytriqs.utility.mpi as mpi
from numpy import array, exp, dot, sqrt, pi, log, linspace
from numpy.linalg import norm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ..lattice.superlatticetools import dispersion as energy_dispersion, _init_k_sum, reciprocal_latticevectors

# TODO make plots nice, what if cluster of lattice that is not fully trans inv?
# adjust plots for non-orthogonal reciprocal lattice vectors
# TODO FFT, izip from itertools, mpi

class PeriodizationBase(object):
    """
    Assumes full translation symmetry, 1 band.
    Tested for a square lattice.
    Some plotmethods are written for 2d-lattices only, the rest is supposed to be generic.
    """

    def __init__(self, lattice_vectors = [[0, 0, 0]], lattice_basis = [[0, 0, 0]], hopping = {(0, 0) : [[0]]}, n_kpts = 0, superlattice_basis = [[0]], archive = False):
        """
        The lattice properties have to be those of the FULLY periodized Lattice, i.e. not those of any superlattice.
        coordinates-basis:
        lattice_vectors : cartesian
        lattice_basis : lattice
        hopping : lattice
        superlattice_basis: lattice
        """
        if archive:
            self.load_from_disk(archive)
        else:
            self.n_kpts = n_kpts
            self.hopping = hopping
            self.lattice_basis = lattice_basis
            self.lattice_vectors = lattice_vectors
            self.superlattice_basis = superlattice_basis
            sumk = _init_k_sum(lattice_vectors, lattice_basis, hopping, n_kpts)
            #self.bz_grid = sumk.BZ_Points
            # TODO check gridfix for other lattice structures than squarelattice
            shift = sumk.BZ_Points[1, 0] - sumk.BZ_Points[0, 0]
            d = len(sumk.BZ_Points[0, :])
            self.d = d
            #self.bz_grid = array([(k - array([shift] * d)) for k in sumk.BZ_Points]) # only valid for 2x2cluster in squarelattice!
            self.bz_grid = array([(k - shift) for k in sumk.BZ_Points])
            self.bz_grid = array([k for k in sumk.BZ_Points]) # remove shift
            self.bz_weights = sumk.BZ_weights
            self.reciprocal_lattice_vectors = reciprocal_latticevectors(lattice_vectors)
            self.eps = sumk.Hopping
            #self.sumk = sumk

    def get_sigma_lat(self):
        return self.sigma_lat

    def get_g_lat(self):
        return self.g_lat

    def get_g_lat_loc(self):
        return self.g_lat_loc

    def set_g_lat_loc(self, g_lat):
        self.g_lat_loc = _g_lat_loc(g_lat, self.bz_weights, self.d)

    def get_tr_g_lat(self):
        return self.tr_g_lat

    def set_tr_g_lat(self, g_lat):
        self.tr_g_lat = _tr_g_lat(g_lat)

    def get_sigma_lat_loc(self):
        return self.sigma_lat_loc

    def set_sigma_lat_loc(self, sigma_lat):
        self.sigma_lat_loc = _g_lat_loc(sigma_lat, self.bz_weights, self.d)
        self.sigma_lat_loc.name = '$\Sigma_{lat,loc}$'

    def get_tr_g_lat_pade(self):
        return self.tr_g_lat_pade

    def set_tr_g_lat_pade(self, g_lat, **kwargs):
        self.tr_g_lat_pade = _tr_g_lat_pade(g_lat, **kwargs)

    def get_bz_grid(self):
        return self.bz_grid

    def cartesian_vector(self, vector):
        v = array(vectors)
        t = array(self.lattice_vectors)
        return dot(t, v)

    def cartesian_matrix(self, matrix):
        m = array(matrix)
        t = array(self.lattice_vectors)
        return dot(m, t.T)

    def write_to_disk(self, filename):
        arch = HDFArchive(filename, 'a')
        if arch.is_group('Periodization'):
            del arch['Periodization']
        arch.create_group('Periodization')
        a_p = arch['Periodization']
        for key, val in self.__dict__.items():
            if key == 'hopping':
                a_p.create_group(key)
                a_h = a_p[key]
                for r, h in val.items():
                    a_h.create_group(str(r))
                    a_h_r = a_h[str(r)]
                    a_h_r['R'] = r
                    a_h_r['h'] = h
            elif type(val) == list:
                saves = dict([(str(i), val[i]) for i in range(len(val))])
                a_p.create_group(key)
                a_p[key].update(saves)
            elif type(val) == dict:
                a_p.create_group(key)
                a_p[key].update(val)                
            else:
                a_p[key] = val
        del arch

    def load_from_disk(self, filename):
        arch = HDFArchive(filename, 'r')
        a_p = arch['Periodization']
        for key, val in a_p.items():
            if key == 'hopping':
                self.__dict__[key] = dict()
                for r in val.keys():
                    self.__dict__[key].update({val[r]['R'] : val[r]['h']})
            elif a_p.is_group(key):
                if type(val) == BlockGf:
                    self.__dict__[key] = val
                elif type(val) == dict:
                    if key != 'hopping':
                        self.__dict__[key] = dict()
                        self.__dict__[key].update(val)
                    else:
                        pass
                else:
                    self.__dict__[key] = [0] * len(a_p[key])
                    for key2, val2 in val.items():
                        self.__dict__[key][int(key2)] = val2
            elif a_p.is_data(key):
                self.__dict__[key] = val

    def plot_dos_loc(self, **kwargs):
        k_dos = ['pade_n_omega_n', 'pade_eta', 'dos_n_points', 'dos_window']
        p_dos = dict()
        for key, val in kwargs.items():
            if key in k_dos:
                p_dos.update({key : val})
                del kwargs[key]
        if not 'name' in kwargs:
            kwargs.update({'name' : 'LDOS'})
        oplot(_tr_g_lat_pade([self.g_lat_loc], **p_dos)[0], RI = 'S', **kwargs)

    def plot_hist2d_energy_dispersion(self, band = 0, **kwargs):
        """
        Written only for 2d-lattice data
        """
        #plot_grid = list()
        #for k in self.bz_grid:
        #    plot_grid.append(dot(self.reciprocal_lattice_vectors, k))
        #plot_grid = array(plot_grid)
        #k1 = self.reciprocal_lattice_vectors[0, :]
        #k2 = self.reciprocal_lattice_vectors[1, :]
        #nbins = int(self.n_kpts * (abs(dot(k1, k2)) + norm(k1)) / norm(k1))
        eps = [self.eps[k, band, band].real for k in range(len(self.eps))]
        plt.hist2d(self.bz_grid[:, 0], self.bz_grid[:, 1], bins = self.n_kpts, weights = eps, **kwargs)
        plt.colorbar()
        plt.xlabel('$k_x$')
        plt.ylabel('$k_y$')
        plt.title('$\epsilon_' + str(band) + '(k)$')

    def plot_hist2d_dos_k(self, ind_freq = 'ind_zerofreq', logarithmic = False, **kwargs):
        if ind_freq == 'ind_zerofreq':
            ind_freq = int(len(self.tr_g_lat_pade[0].data[:, 0, 0]) * .5)
        if logarithmic:
            plt.hist2d(self.bz_grid[:, 0], self.bz_grid[:, 1], bins = self.n_kpts, weights = [log(-1 * self.tr_g_lat_pade[k].data[ind_freq, 0, 0].imag / pi) for k in range(len(self.tr_g_lat_pade))], **kwargs)
        else:
            #plt.hist2d(self.bz_grid[:, 0], self.bz_grid[:, 1], bins = self.n_kpts, weights = [(+1 * self.tr_g_lat_pade[k].data[ind_freq, 0, 0].imag / pi) for k in range(len(self.tr_g_lat_pade))], cmin = -0.1, vmin = 0, vmax = 1, **kwargs)
            plt.hist2d(self.bz_grid[:, 0], self.bz_grid[:, 1], bins = self.n_kpts, weights = [(-1 * self.tr_g_lat_pade[k].data[ind_freq, 0, 0].imag / pi) for k in range(len(self.tr_g_lat_pade))], **kwargs)
        plt.colorbar()
        plt.xlabel('$k_x$')
        plt.ylabel('$k_y$')
        plt.title('$A(k, \omega)$')

    def plot_dos_k_w(self, path):
        f = self.get_tr_g_lat_pade()
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        n_omega = len(f[0].data[:, 0, 0])
        k_ticks = list()

        for nr_k, k_ind in enumerate(_k_ind_path(self.bz_grid, path)):
            for p in path:
                if all(self.bz_grid[k_ind] == _k(self.bz_grid, p)):
                    k_ticks.append([nr_k, p])
            x = range(n_omega)
            y = [nr_k] * n_omega
            z = -f[k_ind].data[:, 0, 0].imag / pi
            ax.plot(x, y, z, label = 'test', c = 'b')

        ax.set_xlabel('$\omega$')
        ax.set_xticks([int(n_omega * .5)])
        ax.set_xticklabels([0])
        ax.set_xlim([0, n_omega])
        ax.set_ylabel('$k$')
        ax.set_yticks([k_ticks[i][0] for i in range(len(k_ticks))])
        ax.set_yticklabels([k_ticks[i][1] for i in range(len(k_ticks))])
        ax.set_zlabel('$A(k, \omega)$')
        #ax.set_zlim([0, 4])
        ax.view_init(elev = 60, azim = -90)

    def plot(self, function_name, k, block, index, *args, **kwargs):
        """
        makes a 1D plot of function_name: 'M_lat', 'Sigma_lat' or 'G_lat'
        """
        assert function_name in ['M_lat', 'Sigma_lat', 'G_lat'], 'invalid function_name'
        if function_name == 'M_lat': function = self.m_lat
        if function_name == 'Sigma_lat': function = self.sigma_lat
        if function_name == 'G_lat': function = self.g_lat
        pplot(function, k, block, index, self.bz_grid, *args, **kwargs)

    def plot2d_k(self, function_name, matsubara_freq = 0, spin = 'up', band = 0, **kwargs):
        """
        makes a 2D histogramplot of function_name: 'M_lat', 'Sigma_lat', 'G_lat' or 'Tr_G_lat' over the k vectors of the 1BZ
        """
        assert function_name in ['M_lat', 'Sigma_lat', 'G_lat', 'Tr_G_lat'], 'invalid_function_name'
        if function_name == 'M_lat': 
            pname = '$M_{lat,' + str(band) + '}$_' + str(spin)
            function = self.m_lat
            pplot_hist2d_k(function, spin, matsubara_freq, band, self.bz_grid, self.n_kpts, **kwargs)
        if function_name == 'Sigma_lat': 
            pname = '$\Sigma_{lat,' + str(band) + '}$_' + str(spin)
            function = self.sigma_lat
            pplot_hist2d_k(function, spin, matsubara_freq, band, self.bz_grid, self.n_kpts, **kwargs)
        if function_name == 'G_lat':
            pname = '$G_{lat,' + str(band) + '}$_' + str(spin)
            function = self.g_lat
            pplot_hist2d_k(function, spin, matsubara_freq, band, self.bz_grid, self.n_kpts, **kwargs)
        if function_name == 'Tr_G_lat': 
            pname = 'Tr$G_{lat}$'
            function = self.tr_g_lat
            pplot_hist2d_k_singleG(function, matsubara_freq, 0, self.bz_grid, self.n_kpts, **kwargs)
        if 'imaginary_part' in kwargs:
            if kwargs['imaginary_part'] == False:
                pname = 'Re' + pname
            else:
                pname = 'Im' + pname
        else:
            pname = 'Im' + pname

        plt.xlabel('$k_x$')
        plt.ylabel('$k_y$')
        plt.title(pname + '$(k, i\omega_' + str(matsubara_freq) + ')$')

def _tr_g_lat_pade(g_lat, pade_n_omega_n = 101, pade_eta = 10**(-2), dos_n_points = 1200, dos_window = (-10, 10)):
    tr_g_lat_pade = list()
    for gk in g_lat:
        tr_spin_g = GfImFreq(indices = range(len(gk['up'].data[0, :, :])), mesh = gk.mesh)
        tr_spin_g << gk['up'] + gk['down']
        tr_band_g = GfImFreq(indices = [0], mesh = gk.mesh)
        tr_band_g.zero()
        _temp = tr_band_g.copy()
        for i in range(len(tr_spin_g.data[0, :, :])):
            tr_band_g << _temp + tr_spin_g[i, i]
            _temp << tr_band_g
        del _temp

        tr_g = tr_band_g
        tr_g_lat_pade.append(GfReFreq(indices = [0], window = dos_window,
                                      n_points = dos_n_points, name = 'Tr$G_{lat}$'))
        tr_g_lat_pade[-1].set_from_pade(tr_g, n_points = pade_n_omega_n,
                                        freq_offset = pade_eta)
    return tr_g_lat_pade

def _k_ind(bz_grid, k):
    assert len(k) == len(bz_grid[0]), 'k must have dimension ' + str(len(bz_grid[0]))
    closest_dist = 10**3
    for k_ind, k_grid in enumerate(bz_grid):
        dist = norm(array(k_grid) - array(k))
        if dist < closest_dist:
            ind = k_ind
            closest_dist = dist
    return ind

def _k(bz_grid, k):
    return bz_grid[_k_ind(bz_grid, k)]

def _g_lat_loc(g_lat, bz_weights, d):
    g_lat_loc = g_lat[0].copy()
    g_lat_loc.zero()
    g_lat_loc.name = '$G_{lat,loc}$'
    for i in range(len(g_lat)):
        for n, b in g_lat_loc:
            _temp = g_lat_loc.copy()
            g_lat_loc[n] << _temp[n] + g_lat[i][n] * bz_weights[i]
    del _temp
    return g_lat_loc

def _tr_g_lat(g_lat):
    for s, b in g_lat[0]:
        bands = range(len(b.data[0, :, :]))
        break
    tr_g_lat = list()
    for g_lat_k in g_lat:
        tr_g_lat_k = GfImFreq(indices = [0], mesh = g_lat_k.mesh)
        for spin, block in g_lat_k:
            for band in bands:
                _temp = tr_g_lat_k.copy()
                tr_g_lat_k << _temp + g_lat_k[s][band, band]
        tr_g_lat.append(tr_g_lat_k)
    return tr_g_lat

def _k_ind_path(bz_grid, path, samp_steps = 10**2):
    assert len(path) > 1, 'path must have at least two points'
    inds = [_k_ind(bz_grid, path[0])]
    samp_pos = array(path[0])
    for nr_p in range(1, len(path)):
        for i in range(samp_steps):
            samp_pos = array(path[nr_p - 1]) + (array(path[nr_p]) - array(path[nr_p - 1])) * i / float(samp_steps - 1)
            samp_ind = _k_ind(bz_grid, samp_pos)
            if samp_ind != inds[-1]:
                inds.append(samp_ind)
    return inds

def pplot(f, k, block, index, bz_grid, *args, **kwargs):
    """
    f is assumed to be a list of BlockGf dependent on the k-vectors in bz_grid. k_ind is the index of the k in bz_grid of f(k) to be plotted. kwargs go to oplot.
    """
    if 'name' in kwargs.keys():
        name = kwargs.pop('name')
    else:
        name = str(index[0]) + '_' + str(block) + '_' + str(k)
    oplot(f[_k_ind(bz_grid, k)][block][index], name = name, *args, **kwargs)

def pplot_hist2d_k(f, spin, matsubara_freq, band, bz_grid, n_kpts, imaginary_part = True, *args, **kwargs):
    """
    f is assumed to be a list of BlockGf dependent on the k-vectors in k. kwargs go to hist2d. Written for 2d.
    """
    assert len(bz_grid[0, :]) == 2, 'Data is not from a 2d calculation'
    if imaginary_part:
        plt.hist2d(bz_grid[:, 0], bz_grid[:, 1], bins = n_kpts, weights = [f[k][spin].data[matsubara_freq, band, band].imag for k in range(len(f))], *args, **kwargs)
    else:
      plt.hist2d(bz_grid[:, 0], bz_grid[:, 1], bins = n_kpts, weights = [f[k][spin].data[matsubara_freq, band, band].real for k in range(len(f))], *args, **kwargs)
    plt.colorbar()

def pplot_hist2d_k_singleG(f, matsubara_freq, band, bz_grid, n_kpts, imaginary_part = True, *args, **kwargs):
    """
    f is assumed to be a list of GfImFreq dependent on the k-vectors in k. kwargs go to hist2d. Written for 2d.
    """
    assert len(bz_grid[0, :]) == 2, 'Data is not from a 2d calculation'
    if imaginary_part:
        plt.hist2d(bz_grid[:, 0], bz_grid[:, 1], bins = n_kpts, weights = [f[k].data[matsubara_freq, band, band].imag for k in range(len(f))], *args, **kwargs)
    else:
      plt.hist2d(bz_grid[:, 0], bz_grid[:, 1], bins = n_kpts, weights = [f[k].data[matsubara_freq, band, band].real for k in range(len(f))], *args, **kwargs)
    plt.colorbar()
