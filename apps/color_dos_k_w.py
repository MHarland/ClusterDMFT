#!/usr/bin/env pytriqs
from ClusterDMFT.periodization import PeriodizationBase as Periodization
from matplotlib import pyplot as plt
import sys

for arch in sys.argv[1:]:
    lat = Periodization(archive = arch)
    lat.color_dos_k_w([[0, 0], [.5, 0], [.5, .5], [0, 0]])
    plt.savefig(arch[:-3] + '_dos_kw.png', dpi = 600)
