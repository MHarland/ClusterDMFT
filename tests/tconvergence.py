import unittest, numpy as np
from pytriqs.gf.local import BlockGf, GfImFreq

from ClusterDMFT.convergence import StandardDeviation, ConvergenceAnalysis, Distance

class TestConvergence(unittest.TestCase):
    def run_StandardDeviation(self):
        block1 = GfImFreq(beta = 10, indices = range(2), n_points = 2)
        block1.data[2,:,:] = np.array([[0,0],[0,0]])
        block1.data[3,:,:] = np.array([[0,0],[0,0]])
        block2 = GfImFreq(beta = 10, indices = range(2), n_points = 2)
        block2.data[2,:,:] = np.array([[0,1],[2,3]])
        block2.data[3,:,:] = np.array([[4,5],[6,complex(7,8)]])
        g1 = BlockGf(name_list = ["up", "dn"], block_list = [block1, block2])
        g2 = BlockGf(name_list = ["up", "dn"], block_list = [block2, block1])
        stddev = StandardDeviation([g1, g2])
        stddev.calc_quantity([2,4])
        self.assertEqual(stddev.data_g["up"].data[3,1,1], complex(3.5, 4))

    def run_Distance(self):
        block1 = GfImFreq(beta = 10, indices = range(2), n_points = 2)
        block1.data[2,:,:] = np.array([[0,0],[0,0]])
        block1.data[3,:,:] = np.array([[0,0],[0,0]])
        block2 = GfImFreq(beta = 10, indices = range(2), n_points = 2)
        block2.data[2,:,:] = np.array([[0,1],[2,3]])
        block2.data[3,:,:] = np.array([[4,5],[6,complex(7,8)]])
        g1 = BlockGf(name_list = ["up", "dn"], block_list = [block1, block2])
        g2 = BlockGf(name_list = ["up", "dn"], block_list = [block2, block1])
        dist = Distance([g1, g2])
        dist.calc_quantity([2,4])
        for blockname, block in dist.data_g:
            if blockname == "dn":
                self.assertEqual(block.data[3,1,1], complex(-7, -8))

    def run_ConvergenceAnalysis(self):
        block1 = GfImFreq(beta = 10, indices = range(2), n_points = 2)
        block1.data[2,:,:] = np.array([[0,0],[0,0]])
        block1.data[3,:,:] = np.array([[0,0],[0,0]])
        block2 = GfImFreq(beta = 10, indices = range(2), n_points = 2)
        block2.data[2,:,:] = np.array([[0,1],[2,3]])
        block2.data[3,:,:] = np.array([[4,5],[6,complex(7,8)]])
        g1 = BlockGf(name_list = ["up", "dn"], block_list = [block1, block2])
        g2 = BlockGf(name_list = ["up", "dn"], block_list = [block2, block1])
        analysis = ConvergenceAnalysis(f_list = [g1, g2])
        analysis.calc_standard_deviations()
        g_dict = analysis.integrated_standard_deviations[1]
        for blockname, block in g_dict.items():
            if blockname == "up":
                self.assertEqual(block[1,1], complex(2.5, 2))
        analysis.calc_distances()
        dist_list = analysis.integrated_distances
        for g_dict in dist_list:
            for blockname, block in g_dict.items():
                if blockname == "up":
                    self.assertEqual(block[0,1], 3)
