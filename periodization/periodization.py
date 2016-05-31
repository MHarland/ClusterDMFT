from itertools import izip
from matplotlib.backends.backend_pdf import PdfPages
from pytriqs.gf.local import BlockGf, GfImFreq, iOmega_n, inverse, GfReFreq
from pytriqs.archive import HDFArchive
from pytriqs.plot.mpl_interface import oplot
from pytriqs.utility import mpi
from numpy import array, exp, dot, sqrt, pi, log, linspace, empty
from numpy.linalg import norm
from matplotlib import pyplot as plt, cm
from mpl_toolkits.mplot3d import Axes3D

from ..lattice.superlatticetools import dispersion as energy_dispersion, _init_k_sum
from ..mpiLists import scatter_list, allgather_list

# TODO make plots nice, what if cluster of lattice that is not fully trans inv?
# adjust plots for non-orthogonal reciprocal lattice vectors
# TODO FFT, izip from itertools, mpi

class ClusterPeriodization(object):
    """
    Generic periodization, provides methods for presenting periodization results, independently of the specific method used (cumulant or self-energy).
    """
    def __init__(self, cluster_lattice = [[0, 0, 0]], cluster = [[0, 0, 0]], hopping = {(0, 0) : [[0]]}, n_kpts = 0, blocks = ['up', 'down'], archive = False, groupname = 'periodization', sym_path = [], sym_path_lbls = [], *args, **kwargs):
        """
        coordinates-basis:
        cluster_lattice : cartesian
        cluster : lattice
        t : lattice
        """
        if archive:
            self.__dict__ = self.load_from_disk(archive, groupname)
        else:
            self.n_kpts = n_kpts
            self.hopping = hopping
            self.cluster = cluster
            self.cluster_lattice = cluster_lattice
            sumk = _init_k_sum(cluster_lattice, cluster, hopping, n_kpts)
            self.bz_grid = sumk.bz_points
            self.d = len(sumk.bz_points[0, :])
            self.bz_weights = sumk.bz_weights
            self.eps = sumk.hopping
            self.spins = blocks
            self.sym_path = sym_path
            self.sym_path_lbls = sym_path_lbls

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
        self.sigma_lat_loc.name = '$\Sigma^{(lat,loc)}$'
        for s, b in self.sigma_lat_loc: b.name = '$\Sigma^{(lat,loc)}_{'+s+'}$'

    def get_tr_sigma_lat(self):
        return self.tr_sigma_lat

    def set_tr_sigma_lat(self, sigma_lat):
        self.tr_sigma_lat = _tr_g_lat(sigma_lat)

    def get_tr_g_lat_pade(self):
        return self.tr_g_lat_pade

    def set_tr_g_lat_pade(self, g_lat, **kwargs):
        self.tr_g_lat_pade = _tr_g_lat_pade(g_lat, spins = self.spins, **kwargs)

    def set_dos_loc(self, g_lat_loc, **kwargs):
        k_dos = ['pade_n_omega_n', 'pade_eta', 'dos_n_points', 'dos_window']
        p_dos = dict()
        for key, val in kwargs.items():
            if key in k_dos:
                p_dos.update({key : val})
        if not 'name' in kwargs:
            kwargs.update({'name' : 'LDOS'})
        self.dos_loc = _tr_g_lat_pade([g_lat_loc], self.spins, **p_dos)[0]

    def get_dos_loc(self):
        return self.dos_loc

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

    def write_to_disk(self, filename, groupname = 'periodization'):
        if mpi.is_master_node():
            arch = HDFArchive(filename, 'a')
            if arch.is_group(groupname):
                del arch[groupname]
            arch.create_group(groupname)
            a_p = arch[groupname]
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
        mpi.barrier()

    def load_from_disk(self, filename, groupname):
        if mpi.is_master_node():
            arch = HDFArchive(filename, 'r')
            a_p = arch[groupname]
            for key, val in a_p.items():
                if key == 'hopping':
                    self.__dict__[key] = dict()
                    for r in val.keys():
                        self.__dict__[key].update({val[r]['R'] : val[r]['h']})
                elif a_p.is_group(key):
                    if key in ['g_lat_loc', 'sigma_lat_loc']:
                        self.__dict__[key] = val
                    elif type(val) == GfReFreq:
                        self.__dict__[key] = val
                    elif type(val) == dict and key != 'hopping':
                        self.__dict__[key] = dict()
                        self.__dict__[key].update(val)
                    elif type(val) == tuple:
                        self.__dict__[key] = list()
                        for v in val:
                            self.__dict__[key].append(v)
                    else:
                        self.__dict__[key] = [0] * len(a_p[key])
                        for key2, val2 in val.items():
                            self.__dict__[key][int(key2)] = val2
                elif a_p.is_data(key):
                    self.__dict__[key] = val
                else: print 'Warning: ', key, 'not loaded from archive'
        self.__dict__ = mpi.bcast(self.__dict__)
        return self.__dict__

    def plot_dos_loc(self, **kwargs):
        oplot(self.dos_loc, RI = 'S', **kwargs)

    def plot2d_energy_dispersion(self, band = 0, **kwargs):
        """
        Written only for 2d-lattice data
        """
        assert len(self.bz_grid[0, :]) == 2, 'Data is not from a 2d calculation'
        nk = int(sqrt(len(self.bz_grid)))
        k_mesh1d = self.bz_grid[0:nk, 0]
        f_pdat = empty([nk, nk])
        k_index = 0
        for i in range(nk):
            for j in range(nk):
                f_pdat[j, i] = self.eps[k_index, band, band].real
                k_index += 1
        fig, ax = plt.subplots()
        im = ax.imshow(f_pdat, cmap = cm.jet, extent = [-0.5, 0.5, -0.5, 0.5], interpolation = 'gaussian', origin = 'lower', **kwargs)
        fig.colorbar(im, ax = ax)
        ax.set_xticks([-.5,-.25,0,.25,.5])
        ax.set_yticks([-.5,-.25,0,.25,.5])
        ax.set_xlabel('$k_x$')
        ax.set_ylabel('$k_y$')
        ax.set_title('$\epsilon_' + str(band) + '(k)$')

    def plot2d_dos_k(self, ind_freq = 'ind_zerofreq', **kwargs):
        if ind_freq == 'ind_zerofreq':
            ind_freq = int(len(self.tr_g_lat_pade[0].data[:, 0, 0]) * .5)
        nk = int(sqrt(len(self.bz_grid)))
        k_mesh1d = self.bz_grid[0:nk, 0]
        a = empty([nk, nk])
        k_index = 0
        for i in range(nk):
            for j in range(nk):
                a[j, i] = -1 * self.tr_g_lat_pade[k_index].data[ind_freq, 0, 0].imag / pi
                k_index += 1
        fig, ax = plt.subplots()
        im = ax.imshow(a, vmin = 0, vmax = 1, cmap = cm.copper, extent = [-0.5, 0.5, -0.5, 0.5], interpolation = 'gaussian')
        fig.colorbar(im, ax = ax)
        ax.set_xticks([-.5,-.25,0,.25,.5])
        ax.set_yticks([-.5,-.25,0,.25,.5])
        ax.set_xlabel('$k_x$')
        ax.set_ylabel('$k_y$')
        ax.set_title('$A(k, \omega)$')

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
            x = [w.real for w in f[0].mesh]
            y = [nr_k] * n_omega
            z = -f[k_ind].data[:, 0, 0].imag / pi
            ax.plot(x, y, z, label = 'test', c = 'b')

        ax.set_xlabel('$\omega$')
        ax.set_ylabel('$k$')
        ax.set_yticks([k_ticks[i][0] for i in range(len(k_ticks))])
        ax.set_yticklabels([k_ticks[i][1] for i in range(len(k_ticks))])
        ax.set_zlabel('$A(k, \omega)$')
        ax.set_zlim([0, None])
        ax.view_init(elev = 60, azim = -90)
        plt.tight_layout()

    def color_dos_k_w(self, path, path_labels):
        x, y, z, k_ticks = g_k_to_imshow_data(self.get_tr_g_lat_pade(), path, self.bz_grid)
        fig, ax = plt.subplots()
        im = ax.imshow(z.T, cmap = cm.copper, interpolation = 'gaussian', origin = 'lower', extent = [0, len(x), y[0], y[-1]], vmin = 0, vmax = min(2, z.max()))
        ax.set_ylabel('$\omega$')
        ax.set_xlabel('$k$')
        ax.set_xticks([k_ticks[i][0] for i in range(len(k_ticks))])
        ax.set_xticklabels(path_labels)
        ax.set_title('$A(k,\,\omega)$')
        fig.colorbar(im, ax = ax)
        ext = im.get_extent()
        ax.set_aspect(abs((ext[0] - ext[1])/float(ext[2] - ext[3])))
        plt.tight_layout()

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
        makes a 2D plot of function_name: 'M_lat', 'Sigma_lat', 'G_lat' or 'Tr_G_lat' over the k vectors of the 1BZ
        """
        assert function_name in ['M_lat', 'Sigma_lat', 'G_lat', 'Tr_G_lat', 'Tr_Sigma_lat'], 'invalid_function_name'
        if function_name == 'M_lat': 
            pname = '$M_{lat,' + str(band) + '}$_' + str(spin)
            function = self.m_lat
            pplot2d_k(function, spin, matsubara_freq, band, self.bz_grid, self.n_kpts, **kwargs)
        if function_name == 'Sigma_lat': 
            pname = '$\Sigma_{lat,' + str(band) + '}$_' + str(spin)
            function = self.sigma_lat
            pplot2d_k(function, spin, matsubara_freq, band, self.bz_grid, self.n_kpts, **kwargs)
        if function_name == 'G_lat':
            pname = '$G_{lat,' + str(band) + '}$_' + str(spin)
            function = self.g_lat
            pplot2d_k(function, spin, matsubara_freq, band, self.bz_grid, self.n_kpts, **kwargs)
        if function_name == 'Tr_G_lat': 
            pname = 'Tr$G_{lat}$'
            function = self.tr_g_lat
            pplot2d_k_singleG(function, matsubara_freq, 0, self.bz_grid, self.n_kpts, **kwargs)
        if function_name == 'Tr_Sigma_lat': 
            pname = 'Tr$\Sigma_{lat}$'
            function = self.tr_sigma_lat
            pplot2d_k_singleG(function, matsubara_freq, 0, self.bz_grid, self.n_kpts, **kwargs)
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

    def export_results(self, filename = 'periodization.pdf', prange = (0, 60)):
        """
        routine to export easily a selection of results into a pdf file
        """
        if mpi.is_master_node(): self._export_results(filename = filename, prange = prange)

    def _export_results(self, filename = 'periodization.pdf', prange = (0, 60)):
        #if lat.d == 2: 
        #    path = [[0, 0], [.5, 0], [.5, .5], [0, 0]]
        #    path_labels = ['$\Gamma$', 'X', 'M', '$\Gamma$']
        #    #path = [[0,0],[.5,0],[1/3.,-1/3.],[0,0]]#kagome
        #    #path_labels = ['$\Gamma$', 'M', 'K','$\Gamma$']#kagome
        #if lat.d == 3: 
        #    path = [[0,0,0],[.5,0,.5],[.5,.25,.75],[3/8.,3/8.,.75],[.5,.5,.5],[0,0,0]]#bcc
        #    path_labels = ['$\Gamma$', 'X', 'W', 'K', 'L', '$\Gamma$']
        pp = PdfPages(filename)
        n_orbs = len(self.hopping.values()[0])
        orbs = range(n_orbs)
        cm = plt.cm.jet
        n_colors = max(1, n_orbs**2 - 1)
        plt.gca().set_color_cycle([plt.cm.jet(i/float(n_colors)) for i in range(n_orbs**2)])
        oplot(self.get_g_lat_loc()['up'], '-+', x_window = prange, RI = 'I')
        pp.savefig()
        plt.close()
        plt.gca().set_color_cycle([plt.cm.jet(i/float(n_colors)) for i in range(n_orbs**2)])
        oplot(self.get_g_lat_loc()['up'], '-+', x_window = prange, RI = 'R')
        pp.savefig()
        plt.close()
        plt.gca().set_color_cycle([plt.cm.jet(i/float(n_colors)) for i in range(n_orbs**2)])
        oplot(self.get_sigma_lat_loc()['up'], '-+', x_window = prange, RI = 'I')
        pp.savefig()
        plt.close()
        plt.gca().set_color_cycle([plt.cm.jet(i/float(n_colors)) for i in range(n_orbs**2)])
        oplot(self.get_sigma_lat_loc()['up'], '-+', x_window = prange, RI = 'R')
        pp.savefig()
        plt.close()
        self.plot_dos_loc()
        pp.savefig()
        plt.close()
        if self.sym_path and self.sym_path_lbls:
            for p in self.sym_path:
                for i in orbs: self.plot('G_lat', p, 'up', (0, i), '-+', x_window = prange, color = cm(i /float(n_orbs - 1)))
                plt.gca().set_ylabel('$G_{lat}(i\omega_n)$')
                pp.savefig()
                plt.close()
            for p in self.sym_path:
                for i in orbs: self.plot('Sigma_lat', p,'up', (0, i), '-+', x_window = prange, color = cm(i /float(n_orbs - 1)))
                plt.gca().set_ylabel('$\Sigma_{lat}(i\omega_n)$')
                pp.savefig()
                plt.close()
            self.plot_dos_k_w(self.sym_path)
            pp.savefig()
            plt.close()
            self.color_dos_k_w(self.sym_path, self.sym_path_lbls)
            pp.savefig()
            plt.close()
        if self.d == 2:
            self.plot2d_k('Tr_G_lat', 0, imaginary_part = True)
            pp.savefig()
            plt.close()
            self.plot2d_k('Tr_Sigma_lat', 0, imaginary_part = True)
            pp.savefig()
            plt.close()
            self.plot2d_dos_k()
            pp.savefig()
            plt.close()
            self.plot2d_energy_dispersion()
            pp.savefig()
            plt.close()
        pp.close()
        print filename, 'ready'

def _tr_g_lat_pade(g_lat, spins, pade_n_omega_n = 40, pade_eta = 5*10**(-2), dos_n_points = 1200, dos_window = (-5, 5)):
    n_k = len(g_lat)
    n_sites = len(g_lat[0][spins[0]].data[0,:,:])
    bands = range(n_sites)
    tr_spin_g = GfImFreq(indices = bands, mesh = g_lat[0][spins[0]].mesh)
    tr_band_g = GfImFreq(indices = [0], mesh = g_lat[0][spins[0]].mesh)
    tr_g_lat_pade = [GfReFreq(indices = [0], window = dos_window, n_points = dos_n_points, name = 'Tr$G_{lat}$') for i in range(n_k)]
    tr_g_lat_pade_p = scatter_list(tr_g_lat_pade)
    for g_lat_k, tr_g_lat_pade_k in izip(scatter_list(g_lat), tr_g_lat_pade_p):
        tr_spin_g << (g_lat_k[spins[0]] + g_lat_k[spins[1]]) / len(spins)
        tr_band_g.zero()
        for i in bands: tr_band_g << tr_band_g + tr_spin_g[i, i] / n_sites
        tr_g = tr_band_g
        tr_g_lat_pade_k.set_from_pade(tr_g, n_points = pade_n_omega_n, freq_offset = pade_eta)
    tr_g_lat_pade = allgather_list(tr_g_lat_pade_p)
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
    n_k = len(g_lat)
    g_lat_loc = g_lat[0].copy()
    g_lat_loc.zero()
    g_lat_loc.name = '$G_{lat,loc}$'
    for s, b in g_lat_loc: b.name = '$G^{(lat,loc)}_{'+s+'}$'
    g_latArray = empty([n_k], dtype = object)
    for i in range(n_k): g_latArray[i] = g_lat[i]
    for weight_k, g_lat_k in izip(*[mpi.slice_array(a) for a in [bz_weights, g_latArray]]):
        for s, b in g_lat_loc:
            b += g_lat_k[s] * weight_k
    g_lat_loc << mpi.all_reduce(mpi.world, g_lat_loc, lambda x, y: x + y)
    return g_lat_loc

def _tr_g_lat(g_lat):
    n_k = len(g_lat)
    for s, b in g_lat[0]:
        bands = range(len(b.data[0, :, :]))
        break
    tr_g_lat = [GfImFreq(indices = [0], mesh = g_lat[0].mesh) for i in range(n_k)]
    tr_g_lat_p = scatter_list(tr_g_lat)
    for g_lat_k, tr_g_lat_k in izip(scatter_list(g_lat), tr_g_lat_p):
        for spin, block in g_lat_k:
            for band in bands:
                tr_g_lat_k += g_lat_k[s][band, band] / len(bands) / g_lat[0].n_blocks
    tr_g_lat = allgather_list(tr_g_lat_p)
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
        name = str(block) + str(index[0]) + str(index[1]) +  '_k' + str(k)
    for m in [('R', 'Re','dashed'), ('I', 'Im','solid')]:
        name2 = m[1] + name
        oplot(f[_k_ind(bz_grid, k)][block][index], name = name2, RI = m[0], linestyle = m[2], marker = None, *args, **kwargs)

def pplot2d_k(f, spin, matsubara_freq, band, bz_grid, n_kpts, imaginary_part = True, *args, **kwargs):
    """
    f is assumed to be a list of BlockGf dependent on the k-vectors in k. kwargs go to imshow. Written for 2d.
    """
    assert len(bz_grid[0, :]) == 2, 'Data is not from a 2d calculation'
    nk = int(sqrt(len(bz_grid)))
    k_mesh1d = bz_grid[0:nk, 0]
    f_pdat = empty([nk, nk])
    k_index = 0
    for i in range(nk):
        for j in range(nk):
            if imaginary_part:
                f_pdat[j, i] = f[k_index][spin].data[matsubara_freq, band, band].imag
            else:
                f_pdat[j, i] = f[k_index][spin].data[matsubara_freq, band, band].real
            k_index += 1
    fig, ax = plt.subplots()
    im = ax.imshow(f_pdat, cmap = cm.jet, extent = [-0.5, 0.5, -0.5, 0.5], interpolation = 'gaussian', **kwargs)
    fig.colorbar(im, ax = ax)
    ax.set_xticks([-.5,-.25,0,.25,.5])
    ax.set_yticks([-.5,-.25,0,.25,.5])
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')

def pplot2d_k_singleG(f, matsubara_freq, band, bz_grid, n_kpts, imaginary_part = True, *args, **kwargs):
    """
    f is assumed to be a list of GfImFreq dependent on the k-vectors in k. kwargs go to imshow. Written for 2d.
    """
    assert len(bz_grid[0, :]) == 2, 'Data is not from a 2d calculation'
    nk = int(sqrt(len(bz_grid)))
    k_mesh1d = bz_grid[0:nk, 0]
    f_pdat = empty([nk, nk])
    k_index = 0
    for i in range(nk):
        for j in range(nk):
            if imaginary_part:
                f_pdat[j, i] = f[k_index].data[matsubara_freq, band, band].imag
            else:
                f_pdat[j, i] = f[k_index].data[matsubara_freq, band, band].real
            k_index += 1
    fig, ax = plt.subplots()
    im = ax.imshow(f_pdat, cmap = cm.jet, extent = [-0.5, 0.5, -0.5, 0.5], interpolation = 'gaussian', **kwargs)
    fig.colorbar(im, ax = ax)
    ax.set_xticks([-.5,-.25,0,.25,.5])
    ax.set_yticks([-.5,-.25,0,.25,.5])
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')

def g_k_to_imshow_data(tr_g_lat_pade_k, path, bz_grid):
    f = tr_g_lat_pade_k
    n_omega = len(f[0].data[:, 0, 0])
    k_ticks = list()
    z = list()
    x = list()
    for nr_k, k_ind in enumerate(_k_ind_path(bz_grid, path)):
        for p in path:
            if all(bz_grid[k_ind] == _k(bz_grid, p)) and not([nr_k, p] in k_ticks):
                k_ticks.append([nr_k, p])
        x.append(bz_grid[k_ind])
        z.append(list())
        for n, wn in enumerate(f[0].mesh):
            z[-1].append(-f[k_ind].data[n, 0, 0].imag / pi)
    x = array(x)
    y = array([w.real for w in f[0].mesh])
    z = array(z)
    return x, y, z, k_ticks
