#!/usr/bin/env pytriqs
from cdmft.cdmft import CDmft
from cdmft.periodization.cumulant_periodization import Periodization
from parameters_l_dim import p

my_cdmft = CDmft(archive = 'l_dim.h5')
my_cdmft.parameters.update(p.copy())
my_cdmft.sync_with_archive()
my_cdmft.run_dmft_loops(10)
my_periodization = Periodization([[1, 0, 0], [0, 200, 0]], [[-.5, .5]], {(1, 0) : [[-1]], (-1, 0) : [[-1]]}, 32, [[.5, 0], [-.5, 0]])
my_periodization.set_all(my_cdmft.load('Sigma_c_iw'))
my_periodization.write_to_disk(my_cdmft.parameters['archive'])
my_cdmft.export_results()
