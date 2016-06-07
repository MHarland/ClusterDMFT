import numpy as np, itertools as itt

class ConvergenceAnalysis:
    
    def __init__(self, cdmft = None, f_str = None, first_loop_nr = 0, f_list = None):
        if f_list is None:
            assert cdmft is not None and f_str is not None, "Initialize with either cdmft and f_str or with f_list"
            self.cdmft = cdmft
            self.f_list = [cdmft.load(f_str, i) for i in range(first_loop_nr, cdmft.next_loop())]
        else:
            self.f_list = f_list
        self.indices = indices_list(self.f_list[0])
        assert len(self.f_list) > 0, "No completed DMFT-loops in cdmft-instance."
        self.integrated_standard_deviations = []
        self.integrated_distances = []

    def _get_arg_range(self, arg_range):
        if arg_range is None:
            n_pts = self.f_list[0][self.indices[0][0]].data.shape[0]
            arg_range = [int(n_pts * .5), n_pts]
        return arg_range

    def calc_standard_deviations(self, arg_range = None):
        """Calculates the standard deviations of function f over dmft-loops as a function of the number of dmft-loops included. It integrates over the argument of f."""
        arg_range = self._get_arg_range(arg_range)
        ie = len(self.f_list)
        for ia in range(ie - 1, -1, -1):
            std_dev = StandardDeviation(self.f_list[ia:ie])
            std_dev.calc_quantity(arg_range)
            self.integrated_standard_deviations.append(
                _integrate_arg(std_dev.data_g, arg_range[1] - arg_range[0])
            )

    def calc_distances(self, arg_range = None):
        arg_range = self._get_arg_range(arg_range)
        for ia in range(len(self.f_list) - 1):
            ie = ia + 2
            dist = Distance(self.f_list[ia:ie])
            dist.calc_quantity(arg_range)
            self.integrated_distances.append(
                _integrate_arg(dist.data_g, arg_range[1] - arg_range[0])
            )

def _integrate_arg(g, norm):
    """integrates arg, e.g. over iwn for G(iwn). Returns dict"""
    integral = {}
    for block_name, block in g:
        integral[block_name] = np.sum(block.data, axis = 0) / float(norm)
    return integral

def indices_list(g):
    inds = []
    for bname, bdata in g:
        inds.append([bname, range(bdata.data.shape[1])])
    return inds

class ListAnalysis:
    
    def __init__(self, f_list):
        self.f_list = f_list
        self.data_g = f_list[0].copy()
        self.data_g.zero()
        self._block_indices = [inds for inds in self.data_g.indices]

    def calc_quantity(self, arg_range):
        for b_ind in self._block_indices:
            sub_indices = [ind for ind in self.f_list[0][b_ind].indices]
            gblock_loops = _merge_loops(self.f_list, b_ind)
            for i, j, x in itt.product(sub_indices, sub_indices, range(*arg_range)):
                sample = gblock_loops[:, x, i, j]
                self.data_g[b_ind].data[x, i, j] = self.analyse(sample.real)
                if (sample.imag != 0).any():
                    self.data_g[b_ind].data[x, i, j] += complex(0, self.analyse(sample.imag))

def _merge_loops(f_list, block):
    """Merges the block-data of a dmft-loop list of BlockGf into a numpy.array"""
    return np.array([g[block].data for g in f_list])

class StandardDeviation(ListAnalysis):

    def analyse(self, x):
        return np.std(x)

class Distance(ListAnalysis):

    def __init__(self, f_list, *args, **kwargs):
        assert len(f_list) == 2, "Distance is only for lists of two defined"
        ListAnalysis.__init__(self, f_list, *args, **kwargs)

    def analyse(self, x):
        return abs(np.diff(x))

class AccuracyAnalysis:
    def __init__(self, cdmft, n_converged):
        pass

