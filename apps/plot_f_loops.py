#!/usr/bin/env pytriqs
from ClusterDMFT.cdmft import CDmft
from pytriqs.plot.mpl_interface import oplot
from matplotlib import pyplot as plt
import sys

functions = ['g_transf_iw']
loops = range(0,3,1)
blocks = ['G','M','X','Y']
indices = [(0, 0), (0,1)]
plotrange = (0, 40)

for arch in sys.argv[1:]:
    x = CDmft(archive = arch)
    for f in functions:
        for index in blocks:
            for index2 in indices:
                for i, c in enumerate(['I', 'R']):
                    plt.subplot(2, 1, 1 + i)
                    if c == 'I': plt.gca().set_title('Imaginary Part')
                    if c == 'R': plt.gca().set_title('Real Part')
                    for n, l in enumerate(loops):
                        oplot(x.load(f, l)[index][index2], RI = c, x_window = plotrange, name = 'it' + str(l), color = plt.cm.jet(n/float(len(loops) - 1)))
                plt.tight_layout()
                plt.savefig(arch[0:-3] + '_' + f + '_' + str(index) + '_' + str(index2[0])+str(index2[1])+'.pdf', dpi = 300)
                plt.close()
