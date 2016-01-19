from ClusterDMFT.transformation.nambuplaquette import NambuPlaquetteTransformation
from ClusterDMFT.transformation.sites import ClustersiteTransformation
from ClusterDMFT.lattice.superlattices import TwoByTwoClusterInSquarelattice, DimerInChain

import unittest, pytriqs, itertools, numpy

class TestTransformations(unittest.TestCase):

    def runDimerTransformation(self):
        lat = DimerInChain()
        beta = 10
        blockstates = range(2)
        n_iw = 1024
        blocks = lat.get_blocks()
        g_iw = pytriqs.gf.local.BlockGf(name_block_generator = [(s, pytriqs.gf.local.GfImFreq(indices = blockstates, beta = beta, n_points = n_iw, name = '$G_{c'+s+'}$')) for s in blocks], name = '$G_c$')
        g_iw[blocks[0]] << numpy.array([[1,2],[2,1]])
        transf = ClustersiteTransformation(lat.get_g_transf_struct_orbital(),
                                           lat.get_transf_orbital(),
                                           beta, n_iw, blocks, blockstates)
        g_iw_k = transf.transform(g_iw)
        g_iw_2 = transf.backtransform(g_iw_k)
        for s, b in g_iw:
            for i, j in itertools.product(blockstates, blockstates):
                for n in range(n_iw):
                    self.assertEqual(g_iw[s].data[n,i,j], g_iw_2[s].data[n,i,j])

    def runNambuTransformation(self):
        beta = 10
        blockstates = range(8)
        n_iw = 1024
        blocks = ['full']
        g_iw = pytriqs.gf.local.BlockGf(name_block_generator = [(s, pytriqs.gf.local.GfImFreq(indices = blockstates, beta = beta, n_points = n_iw, name = '$G_{c'+s+'}$')) for s in blocks], name = '$G_c$')
        g_iw[blocks[0]] << numpy.array([[1,2,2,3,4,5,5,6],
                                        [2,1,3,2,5,4,6,5],
                                        [2,3,1,2,5,6,4,5],
                                        [3,2,2,1,6,5,5,4],
                                        [4,5,5,6,1,2,2,3],
                                        [5,4,6,5,2,1,3,2],
                                        [5,6,4,5,2,3,1,2],
                                        [6,5,5,4,3,2,2,1]])
        lat = TwoByTwoClusterInSquarelattice()
        transf = NambuPlaquetteTransformation(lat.get_g_transf_struct_nambu(),
                                              lat.get_transf_orbital(),
                                              beta, n_iw, blocks, blockstates)
        g_iw_k = transf.transform(g_iw)
        g_iw_2 = transf.backtransform(g_iw_k)
        for s, b in g_iw:
            for i, j in itertools.product(blockstates, blockstates):
                for n in range(n_iw):
                    self.assertEqual(g_iw[s].data[n,i,j], g_iw_2[s].data[n,i,j])
