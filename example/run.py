#!/usr/bin/env pytriqs
from ClusterDMFT.cdmft import CDmft
from ClusterDMFT.periodization.cumulant_periodization import MPeriodization
from parameters_l_dim import p

from pytriqs.gf.local import BlockGf, GfImFreq
import numpy

my_cdmft = CDmft(archive = 'l_dim.h5')
my_cdmft.parameters.update(p.copy())
#my_cdmft.parameters['sigma_c_iw'] = BlockGf(name_list = ['up','down'], block_list = [GfImFreq(indices = range(2), beta = p['beta'])]*2)
#my_cdmft.parameters['sigma_c_iw'] << numpy.array([[1.,1.],[1.,1.]])
my_cdmft.run_dmft_loops(3)
#my_periodization = Periodization([[1, 0, 0], [0, 200, 0]], [[-.5, .5]], {(1, 0) : [[-1]], (-1, 0) : [[-1]]}, 32, [[.5, 0], [-.5, 0]])
#my_periodization.set_all(my_cdmft.load('Sigma_c_iw'))
#my_periodization.write_to_disk(my_cdmft.parameters['archive'])
my_cdmft.export_results()
