from numpy import array, empty, dot, absolute, zeros, sqrt, cos, sin, pi, matrix, kron
from numpy.linalg import norm, inv
from pytriqs.lattice.tight_binding import BravaisLattice, TightBinding, TBLattice
from pytriqs.lattice.lattice_tools import energies_on_bz_grid
from pytriqs.sumk import SumkDiscreteFromLattice

class Superlattice(object):
    """
    Written for the hopping of a supercell, n.n. only
    Assuming that hopping among all sites is equal
    """

    def __init__(self, sl_vec, sl_basis, h):
        """
        sl_vec: superlatticevectors, cartesian coordinates
        sl_basis: superlatticebasis, superlattice coordinates
        h: hopping tensor of the basis, superlattice coordinates
        """
        self.sl_vec = sl_vec
        self.sl_basis = sl_basis
        self.h = h
        self.dimension = len(sl_vec)
        self.h_eff = dict()
        self.w = 0

        translation = empty(3, dtype = int)
        for translation[0] in range(-1, 2):
            if self.dimension == 1:
                self.h_eff.update({translation[0] : self._get_h_el(translation[0])})
            for translation[1] in range(-1, 2):
                if self.dimension == 2:
                    self.h_eff.update({(translation[0], translation[1]) : self._get_h_el(translation[0], translation[1])})
                for translation[2] in range(-1, 2):
                    if self.dimension == 3:
                        self.h_eff.update({(translation[0], translation[1], translation[2]) : self._get_h_el(translation[0], translation[1], translation[2])})
        self.h_eff = drop_zeros(self.h_eff)

        eps = dispersion(self.get_cartesian_superlatticevectors(), self.get_superlatticebasis(), self.get_h_eff(), 64)
        self.w = eps.max() - eps.min()

    def get_h_eff(self):
        return self.h_eff

    def get_cartesian_superlatticevectors(self):
        return self.sl_vec

    def get_cartesian_superlatticebasis(self):
        cb = list()
        for v in self.sl_basis:
            cb.append(self._sl_cart(v))
        return cb

    def get_superlatticebasis(self):
        return self.sl_basis

    def get_bandwidth(self, n_kpts):
        eps = dispersion(self.get_cartesian_superlatticevectors(), self.get_superlatticebasis(), self.get_h_eff(), n_kpts)
        return eps.max() - eps.min()

    def get_eps_min(self, n_kpts):
        return dispersion(self.get_cartesian_superlatticevectors(), self.get_superlatticebasis(), self.get_h_eff(), n_kpts).min()

    def get_eps_max(self, n_kpts):
        return dispersion(self.get_cartesian_superlatticevectors(), self.get_superlatticebasis(), self.get_h_eff(), n_kpts).max()

    def get_u(self, u_w_ratio, n_kpts = False):
        if not n_kpts:
            return u_w_ratio * self.w
        else:
            return u_w_ratio * self.get_bandwidth(n_kpts)

    def get_rounded_u(self, *args, **kwargs):
        return int(round(self.get_u(*args, **kwargs), 0))

    def show_h_eff(self):
        for k in self.h_eff.keys():
            print k
            print self.h_eff[k]
            print

    def _get_h_el(self, *args):
        translation = list()
        almost_zero = 10e-10
        h_el = zeros([len(self.sl_basis)] * 2, dtype = float)
        for i in args:
            translation.append(i)

        for i, r_i in enumerate(self.sl_basis):
            for j, r_j in enumerate(self.sl_basis):
                for k in self.h.keys():
                    if norm(self._sl_cart(translation) + self._sl_cart(r_i) - self._sl_cart(r_j) - self._sl_cart([k])) < almost_zero:
                        h_el[i, j] = self.h[k]
        return h_el

    def _sl_cart(self, vec):
        return dot(array(vec), array(self.sl_vec))

def sum_list(l):
    if not l:
        return 0
    else:
        s = l[0]
        for i in range(1, len(l)):
            s += l[i]
        return s

def drop_zeros(dic_of_arrays):
    for k in dic_of_arrays.keys():
        is_zero = True
        for i in range(len(dic_of_arrays[k])):
            for j in range(len(dic_of_arrays[k])):
                if dic_of_arrays[k][i, j] != 0:
                    is_zero = False
        if is_zero:
            del dic_of_arrays[k]
    return dic_of_arrays

def dispersion(l_vec, l_basis, h, n_kpts):
    l_basis_c = list()
    for v in l_basis:
        l_basis_c.append(cartesian(l_vec, v))
    l_basis_c = array(l_basis_c)
    bravais = BravaisLattice(l_vec, l_basis_c)
    tb = TightBinding(bravais, h)
    return energies_on_bz_grid(tb, n_kpts)

def _init_k_sum(lattice_vectors, lattice_basis, hopping, n_kpts, use_TBSuperLattice = False):
    basis_c = list()
    for v in lattice_basis:
        basis_c.append(cartesian(lattice_vectors, v))
    basis_c = array(basis_c)
    
    if use_TBSuperLattice:
        sublattice = TBLattice(units = basis_c, hopping = hopping)
        lattice = TBSuperLattice(tb_lattice = sublattice, 
                                 super_lattice_units = lattice_vectors)
    else:
        lattice = TBLattice(units = lattice_vectors, hopping = hopping, orbital_positions = basis_c, orbital_names = [str(i) for i in range(len(lattice_basis))])
    return SumkDiscreteFromLattice(lattice = lattice, 
                                   n_points = n_kpts, 
                                   method = 'Riemann')

def cartesian(lat_vec, vector):
    vl = array(lat_vec)
    v = zeros([3])
    for i in range(len(vector)):
        v[i] = vector[i]
    v2 = zeros([3])
    for i in range(len(v)):
        v2[i] = v[i]
    return v2

def reciprocal_latticevectors(lat_vec):
    if len(lat_vec) == 1:
        print 'rec lat not tested'
        a1 = lat_vec[0]
        return 2 * pi * a1 / norm(a1)
    if len(lat_vec) == 2:
        x = array([[1], [0]])
        y = array([[0], [1]])
        a1 = array([[lat_vec[0][0]], [lat_vec[0][1]]])
        a2 = array([[lat_vec[1][0]], [lat_vec[1][1]]])
        b1 = 2 * pi * dot((kron(x, y.T) - kron(y, x.T)).T, a2) / dot(a1.T, dot((kron(x, y.T) - kron(y, x.T)).T, a2))
        b2 = 2 * pi * dot((kron(y, x.T) - kron(x, y.T)).T, a1) / dot(a2.T, dot((kron(y, x.T) - kron(x, y.T)).T, a1))
        return array([[b1[0, 0], b1[1, 0]], [b2[0, 0], b2[1, 0]]])
    if len(lat_vec) == 3:
        print 'rec lat not tested'
        m = matrix(lat_vec)
        m = m.T
        mb = (2 * pi * inv(m)).T
        return mb[:, 0], mb[:, 1], mb [:, 2]
