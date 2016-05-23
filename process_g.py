from itertools import product
import numpy as np

from pytriqs.utility import mpi
from pytriqs.gf.local import BlockGf, inverse

def clip_g(bg, threshold): # TODO tail consistency(?)
    if not type(bg) == BlockGf: return clip_block(bg, threshold)
    for s, g in bg:
        bg[s] = clip_block(g, threshold)
    return bg

def clip_block(g, threshold):
    for i in range(len(g.data[0, :, :])):
        for j in range(len(g.data[0, :, :])):
            clip = True
            for iw in range(len(g.data[:, 0, 0])):
                if abs(g.data[iw, i, j].real) > threshold:
                    clip = False
            if clip:
                for iw in range(len(g.data[:, 0, 0])):
                    g.data[iw, i, j] = complex(0, g.data[iw, i, j].imag)
            clip = True
            for iw in range(len(g.data[:, 0, 0])):
                if abs(g.data[iw, i, j].imag) > threshold:
                    clip = False
            if clip:
                for iw in range(len(g.data[:, 0, 0])):
                    g.data[iw, i, j] = complex(g.data[iw, i, j].real, 0)
    return g
    

def tail_start(g, interval, offset = 4):
    iw_t = -1
    iw_t_i = -1
    for i in range(len(g.data[0, : , :])): # TODO i, j
        iw_t_i_found = False
        iw0 = 0
        while not iw_t_i_found:
            iw_t_i_found = True
            for iw in range(iw0, iw0 + interval - 1):
                if not(abs(g.data[iw + 1, i, i].imag) - abs(g.data[iw, i, i].imag) < 0) or not(abs(g.data[iw + 1, i, i].real) - abs(g.data[iw, i, i].real) < 0):
                    iw_t_i_found = False
            if iw_t_i_found: iw_t_i = iw0 + offset
            iw0 += 1
            if iw0 == len(g.data[:, 0, 0]) - interval:
                print 'No high-freq behavior in interval found'
                iw_t_i_found = True
        if iw_t_i > iw_t: iw_t = iw_t_i
    if iw_t == -1:
        iw_t = len(g.data[:, 0, 0]) - 2
    if mpi.is_master_node(): print 'starting block-tail-fit at iw_n:', iw_t
    return iw_t

def impose_site_symmetries(g, site_symmetries):
    """
    list of lists of tuples : symmetry of classes of site-indices
    example for dimer in a chain:
    site_symmetries = [[(0, 0), (1, 1)], [(1, 0), (0, 1)]]
    """
    g_s = g.copy()
    g_s.zero()
    for sclass in site_symmetries:
        n = len(sclass)
        for selement1 in sclass:
            for selement2 in sclass:
                for spin, block in g:
                    g_s[spin][selement1] += g[spin][selement2] /float(n)
    return g_s

def impose_paramagnetism(g):
    """takes the expectation value of spins"""
    g_s = g.copy()
    g_s.zero()
    for s1, b1 in g:
        for s2, b2 in g:
            g_s[s1] += g[s2] * .5
    return g_s

def impose_afm(g):
    """written for afm on 4 site sublattice (0,3) and (1,2); imposes only on local entries"""
    sA = [0,3]
    sB = [1,2]
    spins = [s for s in g.indices]
    g_s = g.copy()
    for s, b in g_s:
        for i in range(4):
            b[i, i] << 0
    for s1, b1 in g:
        for i, j in product(range(4), range(4)):
            if (i in sA and j in sA) or (i in sB and j in sB):
                s2 = s1
            else:
                for s in spins:
                    if s != s1:
                        s2 = s
                        break
            g_s[s1][i, i] += g[s2][j, j] * .25
    return g_s

class MixUpdate(object):
    def __init__(self, g, mu, x):
        self.g_old = g.copy()
        self.mu_old = mu
        self.x = x

    def __call__(self, g, mu):
        g << self.x * g + (1 - self.x) * self.g_old
        self.g_old = g.copy()
        mu = self.x * mu + (1 - self.x) * self.mu_old
        self.mu_old = mu
        return g, mu

    def set_mix(self, x):
        self.x = x
    def get_mix(self):
        return self.x

def addExtField(g, field):
    if field:
        indices = [i for i in g.indices]
        g = g.copy()
        ginv = g.copy()
        ginv << inverse(g)
        for s, b in ginv:
            for i in range(len(b.data[0,:,:])):
                b[i,i] += field[s][i]
        g << inverse(ginv)
    return g

def setOffdiagsZero(g):
    ginv = g.copy()
    ginv << inverse(g)
    offdiags = ginv.copy()
    for s, b in ginv:
        orbs = range(len(b.data[0,:,:]))
        for i,j in product(*[orbs]*2):
            if i != j:
                b[i, j] << 0
            else:
                offdiags[s][i,j] << 0
    g << inverse(ginv)
    return g, offdiags

def readdOffdiagConstants(g, offdiags):
    ginv = g.copy()
    ginv << inverse(g)
    for s, b in ginv:
        orbs = range(len(b.data[0,:,:]))
        for i,j in product(*[orbs]*2):
            b[i, j] += offdiags[s][i,j]
    g << inverse(ginv)
    return g
