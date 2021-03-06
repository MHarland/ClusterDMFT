from numpy import sqrt, cos, sin, pi, array
from scipy.linalg import block_diag
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
                (1,-1):[[0,0,0,0],[0,0,0,0],[0,s,0,0],[0,0,0,0]]}

    def get_cartesian_clusterlatticevectors(self):
        return [[1, 0, 0], [0, 1, 0]]

    def get_clusterlatticebasis(self):
        return [[0, 0], [0, .5], [.5, 0], [.5, .5]]

    def get_transf_fourier(self):
        return [[.5, .5, .5, .5], [.5, -.5, .5, -.5], [.5, .5, -.5, -.5], [.5, -.5, -.5, .5]]

    def get_transf_C4v(self, phi):
        x = 1/sqrt(2)
        y = .5
        v1 = array([-x,0,0,x])
        v2 = array([0,-x,x,0])
        return [[y]*4,list(cos(phi)*v1-sin(phi)*v2),list(sin(phi)*v1+cos(phi)*v2),[y,-y,-y,y]]

    def get_blocks(self):
        return ['up', 'down']

    def get_blockstates(self):
        return range(4)

    def get_g_transf_struct_fourier(self):
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

    def get_site_symmetries(self):
        return [[(0,0),(3,3),(1,1),(2,2)],
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
                          [0,0,0,0],
                          [0,(1,2),0,0],
                          [0,0,0,0]]}

    def get_checkerboard_symmetry_transformation(self):
        n = 1/sqrt(2)
        print 'check transf!!! todo'
        return [[n,0,0,n],[0,n,n,0],[0,-n,n,0],[-n,0,0,n]]

class TwoByTwoClusterInSquarelatticeNambu(TwoByTwoClusterInSquarelattice):
    def get_hopping(self, t = -1, tnnn = 0):
        h = TwoByTwoClusterInSquarelattice.get_hopping(self, t, tnnn)
        return dict([(r, block_diag(array(t),array(t))) for r, t in h.items()])

    def get_blocks(self):
        return ['full']

    def get_blockstates(self):
        return range(8)

    def get_g_transf_struct_fourier(self):
        return [[i, range(2)] for i in ['G','X','Y','M']]

    def get_periodization_afm(self):
        p = TwoByTwoClusterInSquarelattice.get_periodization_afm(self)
        new_p =  dict()
        for r, s in p.items():
            new_p[r] = []
            for i in range(4):
                new_p[r].append([])
                for j in range(4):
                    new_p[r][i].append(p[r][i][j])
                for j in range(4):
                    new_p[r][i].append(0)
            for i in range(4,8):
                new_p[r].append([])
                for j in range(4):
                    new_p[r][i].append(0)
                for j in range(4):
                    new_p[r][i].append(p[r][i-4][j])
        return new_p

class TwoByTwoClusterInSquarelatticeNambuRotated(TwoByTwoClusterInSquarelatticeNambu, TwoByTwoClusterInSquarelattice):
    def get_hopping(self, t = -1, tnnn = 0):
        h = TwoByTwoClusterInSquarelattice.get_hopping(self, t, tnnn)
        hh = dict()
        r = 1
        s = 1
        r_map = {(0,0):(0,0),(1,-1):(r,0),(1,0):(s,s),(1,1):(0,r),(0,1):(-s,s),(-1,1):(-r,0),(-1,0):(-s,-s),(-1,-1):(0,-r),(0,-1):(s,-s)}
        for r, hmatrix in h.items():
            hh[r_map[r]] = hmatrix
        return dict([(r, block_diag(array(t),array(t))) for r, t in hh.items()])

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
                (1,-1):[[0,0,0,0],[0,0,0,0],[0,s,0,0],[0,0,0,0]]}

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
                  (1,-1):[[0,0,0,0],[0,0,0,0],[0,s,0,0],[0,0,0,0]]})
        return h

class Pyrochlore(object):
    def __init__(self):
        pass

    def get_hopping(self, t1 = -1., t2 = 0, t3 = 0, t4 = 0):
        return {(-1,1,0) :[[t3,0.,t2,t4],[t2,t3,t1,t2],[0.,0.,t3,0.],[t4,0.,t2,t3]],
                (1,0,0)  :[[t3,t1,t2,t2],[0.,t3,0.,0.],[0.,t2,t3,t4],[0.,t2,t4,t3]],
                (0,0,-1) :[[t3,0.,0.,0.],[t2,t3,t4,0.],[t2,t4,t3,0.],[t1,t2,t2,t3]],
                (0,1,-1) :[[t3,t4,t2,0.],[t4,t3,t2,0.],[0.,0.,t3,0.],[t2,t2,t1,t3]],
                (-1,1,-1):[[0.,0.,0.,0.],[t4,0.,t4,0.],[0.,0.,0.,0.],[t4,0.,t4,0.]],
                (-1,0,1) :[[t3,0.,t4,t2],[t2,t3,t2,t1],[t4,0.,t3,t2],[0.,0.,0.,t3]],
                (-1,0,0) :[[t3,0.,0.,0.],[t1,t3,t2,t2],[t2,0.,t3,t4],[t2,0.,t4,t3]],
                (1,0,-1) :[[t3,t2,t4,0.],[0.,t3,0.,0.],[t4,t2,t3,0.],[t2,t1,t2,t3]],
                (-1,-1,1):[[0.,0.,0.,0.],[t4,0.,0.,t4],[t4,0.,0.,t4],[0.,0.,0.,0.]],
                (1,-1,-1):[[0.,0.,0.,0.],[0.,0.,0.,0.],[t4,t4,0.,0.],[t4,t4,0.,0.]],
                (1,-1,1) :[[0.,t4,0.,t4],[0.,0.,0.,0.],[0.,t4,0.,t4],[0.,0.,0.,0.]],
                (0,0, 1) :[[t3,t2,t2,t1],[0.,t3,t4,t2],[0.,t4,t3,t2],[0.,0.,0.,t3]],
                (0,-1,1) :[[t3,t4,0.,t2],[t4,t3,0.,t2],[t2,t2,t3,t1],[0.,0.,0.,t3]],
                (-1,1,1) :[[0.,0.,t4,t4],[0.,0.,t4,t4],[0.,0.,0.,0.],[0.,0.,0.,0.]],
                (1,-1,0) :[[t3,t2,0.,t4],[0.,t3,0.,0.],[t2,t1,t3,t2],[t4,t2,0.,t3]],
                (1,1,-1) :[[0.,t4,t4,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,t4,t4,0.]],
                (0,1,0)  :[[t3,t2,t1,t2],[0.,t3,t2,t4],[0.,0.,t3,0.],[0.,t4,t2,t3]],
                (0,0,0)  :[[0.,t1,t1,t1],[t1,0.,t1,t1],[t1,t1,0.,t1],[t1,t1,t1,0.]],
                (0,-1,0) :[[t3,0.,0.,0.],[t2,t3,0.,t4],[t1,t2,t3,t2],[t2,t4,0.,t3]]}

    def get_cartesian_clusterlatticevectors(self):
        return [[1, 0, 0],[.5, .5 * sqrt(3), 0],[.5, .5 * 1/sqrt(3), sqrt(2. / 3)]]

    def get_clusterlatticebasis(self):
        return [[0, 0, 0], [.5, 0, 0], [0, .5, 0], [0, 0, .5]]

    def get_blockstates(self):
        return range(4)

    def get_blocks(self):
        return ['up', 'down']

    def get_transf_fourier(self):
        return [[.5, .5, .5, .5], [.5, -.5, .5, -.5], [.5, .5, -.5, -.5], [.5, -.5, -.5, .5]]

    def get_transf_dimer(self):
        n = 1/sqrt(2)
        return [[n,-n,0,0],[n,n,0,0],[0,0,n,n],[0,0,-n,n]]

    def get_g_transf_struct_fourier(self):
        return [[s+'-'+str(i), [0]] for s in ['up', 'down'] for i in range(4)]

    def get_g_transf_struct_site(self):
        return [[s, range(4)] for s in ['up', 'down']]

    def get_transf_T(self, phi, theta):
        x = 1/sqrt(2)
        y = .5
        v1 = array([-x,0,0,x])
        v2 = array([0,-x,x,0])
        v3 = array([y,-y,-y,y])
        return [[y]*4,list(cos(theta)*v1+sin(phi)*sin(theta)*v2-cos(phi)*sin(theta)*v3),list(0*v1+cos(phi)*v2+sin(phi)*v3),list(sin(theta)*v1-sin(phi)*cos(theta)*v2+cos(phi)*cos(theta)*v3)]

class DimerInChain(Superlattice):
    def __init__(self):
        pass

    def get_hopping(self, t = -1):
        return {(0, 0) : [[0,t],[t,0]], (1, 0) : [[0,t],[0,0]], (-1, 0) : [[0,0],[t,0]]}

    def get_transf_fourier(self):
        x = 1/sqrt(2)
        return [[x, x], [x, -x]]

    def get_cartesian_clusterlatticevectors(self):
        return [[1, 0, 0], [0, 1, 0]]

    def get_clusterlatticebasis(self):
        return [[0, 0], [.5, 0]]

    def get_blocks(self):
        return ['up', 'dn']

    def get_blockstates(self):
        return range(2)

    def get_g_transf_struct_fourier(self):
        return [[s+'-'+i, [0]] for s in ['up', 'dn'] for i in ['G','X']]

    def get_g_transf_struct_site(self):
        return [[s, range(2)] for s in ['up', 'dn'] for i in range(1)]


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
