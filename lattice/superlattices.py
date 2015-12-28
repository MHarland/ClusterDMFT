from numpy import sqrt, cos, sin, pi, array
from .superlatticetools import Superlattice

class TwoByTwoClusterInSquarelattice(object):
    def __init__(self):
        pass

    def get_hopping(self, t = -1., tnnn = 0):
        s = tnnn
        return {(0,0):[[0,t,t,s],[t,0,s,t],[t,s,0,t],[s,t,t,0]],
                (1,0):[[0,t,0,s],[0,0,0,0],[0,s,0,t],[0,0,0,0]],
                (1,1):[[0,0,0,s],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                (0,1):[[0,0,t,s],[0,0,s,t],[0,0,0,0],[0,0,0,0]],
                (-1,1):[[0,0,0,0],[0,0,s,0],[0,0,0,0],[0,0,0,0]],
                (-1,0):[[0,0,0,0],[t,0,s,0,],[0,0,0,0],[s,0,t,0]],
                (-1,-1):[[0,0,0,0],[0,0,0,0],[0,0,0,0],[s,0,0,0]],
                (0,-1):[[0,0,0,0],[0,0,0,0],[t,s,0,0],[s,t,0,0]],
                (1,-1):[[0,0,0,0],[0,0,s,0],[0,0,0,0],[0,0,0,0]]}

    def get_cartesian_clusterlatticevectors(self):
        return [[1, 0, 0], [0, 1, 0]]

    def get_clusterlatticebasis(self):
        return [[0, 0], [0, .5], [.5, 0], [.5, .5]]

    def get_transf_orbital(self):
        return [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, -0.5, -0.5], [0.5, -0.5, 0.5, -0.5], [0.5, -0.5, -0.5, 0.5]]

    def get_transf_C4(self, theta):
        x = .5
        phi1 = array([-x,0,0,x])
        phi2 = array([0,-x,x,0])
        phi1p = cos(theta) * phi1 - sin(theta) * phi2
        phi2p = sin(theta) * phi1 + cos(theta) * phi2
        return [[x]*4,list(phi1p),list(phi2p),[x,-x,-x,x]]

    def get_g_transf_struct_orbital(self):
        return [[str(i)+'-'+s, [0]] for s in ['up', 'down'] for i in range(4)]

    def get_g_transf_struct_site(self):
        return [[str(i)+'-'+s, range(4)] for s in ['up', 'down'] for i in range(1)]

    def get_transf_Z2(self):
        x = 1/sqrt(2)
        return [[x,0,x,0],[0,x,0,x],[x,0,-x,0],[0,x,0,-x]]

    def get_site_symmetries_afm(self):
        return [[(0,0),(3,3)],
                [(1,1),(2,2)],
                [(0,1),(1,0),(0,2),(2,0),(1,3),(3,1),(2,3),(3,2)],
                [(0,3),(3,0),(1,2),(2,1)]]

    def get_periodization_afm(self):
        return {(0,0): [[(0,0),(0,1),(0,1),(0,3)],
                        [(0,1),(1,1),(1,2),(0,1)],
                        [(0,1),(1,2),(1,1),(0,1)],
                        [(0,3),(0,1),(0,1),(0,0)]],
                (1,0): [[0,(0,1),0,(0,3)],
                        [0,0,0,0],
                        [0,(1,2),0,(0,1)],
                        [0,0,0,0]],
                (1,1): [[0,0,0,(0,3)],
                        [0,0,0,0],
                        [0,0,0,0],
                        [0,0,0,0]],
                (0,1): [[0,0,(0,1),(0,3)],
                        [0,0,(1,2),(0,1)],
                        [0,0,0,0],
                        [0,0,0,0]],
                (-1,1): [[0,0,0,0],
                         [0,0,(1,2),0],
                         [0,0,0,0],
                         [0,0,0,0]],
                (-1,0): [[0,0,0,0],
                         [(0,1),0,(1,2),0],
                         [0,0,0,0],
                         [(0,3),0,(0,1),0]],
                (-1,-1): [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [(0,3),0,0,0]],
                (0, -1): [[0,0,0,0],
                          [0,0,0,0],
                          [(0,1),(1,2),0,0],
                          [(0,3),(0,1),0,0]],
                (1, -1): [[0,0,0,0],
                          [0,0,(1,2),0],
                          [0,0,0,0],
                          [0,0,0,0]]}

    def get_checkerboard_symmetry_transformation(self):
        n = 1/sqrt(2)
        print 'check transf!!! todo'
        return [[n,0,0,n],[0,n,n,0],[0,-n,n,0],[-n,0,0,n]]

class CheckerboardNNNHopping(TwoByTwoClusterInSquarelattice):
    def get_hopping(self, t = -1., tnnn = 0):
        s = tnnn
        return {(0,0):[[0,t,t,s],[t,0,s,t],[t,s,0,t],[s,t,t,0]],
                (1,0):[[0,t,0,0],[0,0,0,0],[0,0,0,t],[0,0,0,0]],
                (1,1):[[0,0,0,s],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                (0,1):[[0,0,t,0],[0,0,0,t],[0,0,0,0],[0,0,0,0]],
                (-1,1):[[0,0,0,0],[0,0,s,0],[0,0,0,0],[0,0,0,0]],
                (-1,0):[[0,0,0,0],[t,0,0,0,],[0,0,0,0],[0,0,t,0]],
                (-1,-1):[[0,0,0,0],[0,0,0,0],[0,0,0,0],[s,0,0,0]],
                (0,-1):[[0,0,0,0],[0,0,0,0],[t,0,0,0],[0,t,0,0]],
                (1,-1):[[0,0,0,0],[0,0,s,0],[0,0,0,0],[0,0,0,0]]}

class SquareclusterIntegration(TwoByTwoClusterInSquarelattice):
    def get_hopping(self, t = -1., tnnn = 0, alpha = 0, alpha_prime = 0):
        s = tnnn
        h = {(0,0):[[0,t,t,s],[t,0,s,t],[t,s,0,t],[s,t,t,0]]}
        t = alpha * t
        s = alpha_prime * s
        h.update({(1,0):[[0,t,0,s],[0,0,0,0],[0,s,0,t],[0,0,0,0]],
                  (1,1):[[0,0,0,s],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                  (0,1):[[0,0,t,s],[0,0,s,t],[0,0,0,0],[0,0,0,0]],
                  (-1,1):[[0,0,0,0],[0,0,s,0],[0,0,0,0],[0,0,0,0]],
                  (-1,0):[[0,0,0,0],[t,0,s,0,],[0,0,0,0],[s,0,t,0]],
                  (-1,-1):[[0,0,0,0],[0,0,0,0],[0,0,0,0],[s,0,0,0]],
                  (0,-1):[[0,0,0,0],[0,0,0,0],[t,s,0,0],[s,t,0,0]],
                  (1,-1):[[0,0,0,0],[0,0,s,0],[0,0,0,0],[0,0,0,0]]})
        return h

class pyrochlore(object):
    def __init__(self):
        pass

    def get_hopping(self, t = -1.):
        return {(1, 0, 0): array([[ 0., t,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.]]), 
                (0, 0, 1): array([[ 0.,  0.,  0., t],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.]]), 
                (0, -1, 1): array([[ 0.,  0.,  0.,  0.],
                                   [ 0.,  0.,  0.,  0.],
                                   [ 0.,  0.,  0., t],
                                   [ 0.,  0.,  0.,  0.]]), 
                (0, 0, 0): array([[ 0., t, t, t],
                                  [t,  0., t, t],
                                  [t, t,  0., t],
                                  [t, t, t,  0.]]), 
                (-1, 1, 0): array([[ 0.,  0.,  0.,  0.],
                                   [ 0.,  0., t,  0.],
                                   [ 0.,  0.,  0.,  0.],
                                   [ 0.,  0.,  0.,  0.]]), 
                (1, 0, -1): array([[ 0.,  0.,  0.,  0.],
                                   [ 0.,  0.,  0.,  0.],
                                   [ 0.,  0.,  0.,  0.],
                                   [ 0., t,  0.,  0.]]), 
                (1, -1, 0): array([[ 0.,  0.,  0.,  0.],
                                   [ 0.,  0.,  0.,  0.],
                                   [ 0., t,  0.,  0.],
                                   [ 0.,  0.,  0.,  0.]]), 
                (0, 0, -1): array([[ 0.,  0.,  0.,  0.],
                                   [ 0.,  0.,  0.,  0.],
                                   [ 0.,  0.,  0.,  0.],
                                   [t,  0.,  0.,  0.]]), 
                (-1, 0, 1): array([[ 0.,  0.,  0.,  0.],
                                   [ 0.,  0.,  0., t],
                                   [ 0.,  0.,  0.,  0.],
                                   [ 0.,  0.,  0.,  0.]]), 
                (-1, 0, 0): array([[ 0.,  0.,  0.,  0.],
                                   [t,  0.,  0.,  0.],
                                   [ 0.,  0.,  0.,  0.],
                                   [ 0.,  0.,  0.,  0.]]), 
                (0, 1, 0): array([[ 0.,  0., t,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.]]), 
                (0, 1, -1): array([[ 0.,  0.,  0.,  0.],
                                   [ 0.,  0.,  0.,  0.],
                                   [ 0.,  0.,  0.,  0.],
                                   [ 0.,  0., t,  0.]]), 
                (0, -1, 0): array([[ 0.,  0.,  0.,  0.],
                                   [ 0.,  0.,  0.,  0.],
                                   [t,  0.,  0.,  0.],
                                   [ 0.,  0.,  0.,  0.]])}

    def get_cartesian_clusterlatticevectors(self):
        return [[1, 0, 0],[.5, .5 * sqrt(3), 0],[.5, .5 * 1/sqrt(3), sqrt(2. / 3)]]

    def get_clusterlatticebasis(self):
        return [[0, 0, 0], [.5, 0, 0], [0, .5, 0], [0, 0, .5]]

    def get_transf_orbital(self):
        return [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, -0.5, -0.5], [0.5, -0.5, 0.5, -0.5], [0.5, -0.5, -0.5, 0.5]]

    def get_transf_dimer(self):
        n = 1/sqrt(2)
        return [[n,-n,0,0],[n,n,0,0],[0,0,n,n],[0,0,-n,n]]

    def get_g_transf_struct_orbital(self):
        return [[str(i)+'-'+s, [0]] for s in ['up', 'down'] for i in range(4)]

    def get_g_transf_struct_site(self):
        return [[str(i)+'-'+s, range(4)] for s in ['up', 'down'] for i in range(1)]


class l_dim(Superlattice):
    def __init__(self):
        pass

    def get_hopping(self, t = -1):
        return {(1, 0) : [[0,t],[0,0]], (-1, 0) : [[0,0],[t,0]]}

    def get_symmetry_transformation(self):
        return [[1/sqrt(2), 1/sqrt(2)], [1/sqrt(2), -1/sqrt(2)]]

    def get_cartesian_clusterlatticevectors(self):
        return [[1, 0, 0], [0, 1, 0]]

    def get_clusterlatticebasis(self):
        return [[0, 0], [.5, 0]]

    def get_g_transf_struct_orbital(self):
        return [[str(i)+'-'+s, [0]] for s in ['up', 'down'] for i in range(2)]

    def get_g_transf_struct_site(self):
        return [[str(i)+'-'+s, range(2)] for s in ['up', 'down'] for i in range(1)]


"""
class l_3s(Superlattice):
    def __init__(self, t = -1.):
        Superlattice.__init__(self,
                              [[3, 0, 0], [0, 200, 0]], 
                              [[-1 / 3., 0], [0, 0], [1 / 3., 0]], 
                              {(1 / 3., 0) : t, (-1 / 3., 0) : t})

class l_4s(Superlattice):
    def __init__(self, t = -1.):
        Superlattice.__init__(self,
                              [[4, 0, 0], [0, 4, 0]], 
                              [[-3 / 8., 0], [-1 / 8., 0], [1 / 8., 0], [3 / 8., 0]], 
                              {(.25, 0) : t, (-.25, 0) : t})

    def get_C2_transformation(self):
        n = 1/sqrt(2)
        return [[n,0,0,n],[0,n,n,0],[0,-n,n,0],[-n,0,0,n]]


class ss_pyrochlore(Superlattice):
    def __init__(self, t = -1.):
        Superlattice.__init__(self,
                              [[1, 0, 0], 
                               [.5, .5 * sqrt(3), 0], 
                               [.5, 2 / sqrt(3), sqrt(2. / 3)]],
                              [[0, 0, 0]], 
                              {(1, 0, 0) : t, (-1, 0, 0) : t,
                               (0, 1, 0) : t, (0, -1, 0) : t,
                               (0, 0, 1) : t, (0, 0, -1) : t,
                               (1, -1, 0) : t, (-1, 1, 0) : t,
                               (1, 0, -1) : t, (-1, 0, 1) : t,
                               (0, 1, -1) : t, (0, -1, 1) : t})

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

    def get_symmetry_indices(self):
        return [['0-up',[0]],['1-up',[0]],['2-up',[0]],['0-down',[0]],['1-down',[0]],['2-down',[0]]]

class sq_dim(Superlattice):
    def __init__(self, t = -1.):
        Superlattice.__init__(self,
                              [[2, 0, 0], [0, 1, 0]], 
                              [[0, 0], [.5, 0]], 
                              {(.5, 0) : t, (-.5, 0) : t})

    def get_symmetry_transformation(self):
        return [[1/sqrt(2), 1/sqrt(2)], [1/sqrt(2), -1/sqrt(2)]]
"""
