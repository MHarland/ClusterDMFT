from numpy import sqrt, cos, sin, pi

from .superlatticetools import Superlattice

class l_dim(Superlattice):
    def __init__(self, t = -1.):
        Superlattice.__init__(self,
                              [[2, 0, 0], [0, 200, 0]], 
                              [[-.25, 0], [.25, 0]], 
                              {(.5, 0) : t, (-.5, 0) : t})

    def get_symmetry_transformation(self):
        return [[1/sqrt(2), 1/sqrt(2)], [1/sqrt(2), -1/sqrt(2)]]

class l_3s(Superlattice):
    def __init__(self, t = -1.):
        Superlattice.__init__(self,
                              [[3, 0, 0], [0, 200, 0]], 
                              [[-1 / 3., 0], [0, 0], [1 / 3., 0]], 
                              {(1 / 3., 0) : t, (-1 / 3., 0) : t})

class l_4s(Superlattice):
    def __init__(self, t = -1.):
        Superlattice.__init__(self,
                              [[4, 0, 0], [0, 200, 0]], 
                              [[-3 / 8., 0], [-1 / 8., 0], [1 / 8., 0], [3 / 8., 0]], 
                              {(.25, 0) : t, (-.25, 0) : t})

class sq_sq(Superlattice):
    def __init__(self, t = -1.):
        Superlattice.__init__(self,
                              [[2, 0, 0], [0, 2, 0]], 
                              [[0, 0], [.5, 0], [0, .5], [.5, .5]], 
                              {(0, .5) : t, (0, -.5) : t, (.5, 0) : t, (-.5, 0) : t})

    def get_symmetry_transformation(self):
        return [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, -0.5, -0.5], [0.5, -0.5, 0.5, -0.5], [0.5, -0.5, -0.5, 0.5]]

"""
#TODO period shall be 2
class fcc_tet(Superlattice):
    def __init__(self, t = -1.):
        Superlattice.__init__(self,
                              [[1, 0, 0], 
                               [.5, .5 * sqrt(3), 0], 
                               [.5, 2 / sqrt(3), sqrt(2. / 3)]],
                              [[0, 0, 0], [.5, 0, 0], [0, .5, 0], [0, 0, .5]], 
                              {(.5, 0, 0) : t, (-.5, 0, 0) : t,
                               (0, .5, 0) : t, (0, -.5, 0) : t,
                               (0, 0, .5) : t, (0, 0, -.5) : t,
                               (.5, -.5, 0) : t, (-.5, .5, 0) : t,
                               (.5, 0, -.5) : t, (-.5, 0, .5) : t,
                               (0, .5, -.5) : t, (0, -.5, .5) : t})

    def get_symmetry_transformation(self):
        return [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, -0.5, -0.5], [0.5, -0.5, 0.5, -0.5], [0.5, -0.5, -0.5, 0.5]]
"""

class kag_tri(Superlattice):
    def __init__(self, t = -1.):
        Superlattice.__init__(self,
                              [[1, 0, 0], [cos(pi / 3), sin(pi / 3), 0]],
                              [[0, 0], [.5, 0], [0, .5]],
                              {(.5, 0) : t, (-.5, 0) : t,
                               (0, .5) : t, (0, -.5) : t,
                               (-.5, .5) : t, (.5, -.5) : t})

    def get_symmetry_transformation(self):
        return [[1/sqrt(3), 1/sqrt(3), 1/sqrt(3)], [-2/sqrt(6), 1/sqrt(6), 1/sqrt(6)], [0, 1/sqrt(2), -1/sqrt(2)]]

class sq_dim(Superlattice):
    def __init__(self, t = -1.):
        Superlattice.__init__(self,
                              [[2, 0, 0], [0, 1, 0]], 
                              [[0, 0], [.5, 0]], 
                              {(.5, 0) : t, (-.5, 0) : t})

    def get_symmetry_transformation(self):
        return [[1/sqrt(2), 1/sqrt(2)], [1/sqrt(2), -1/sqrt(2)]]
