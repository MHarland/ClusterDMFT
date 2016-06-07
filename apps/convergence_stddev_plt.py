#!/usr/bin/env pytriqs
import matplotlib, sys, itertools as itt
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from ClusterDMFT.convergence import ConvergenceAnalysis
from ClusterDMFT.cdmft import CDmft

function = sys.argv[1]

for fname in sys.argv[2:]:
    cdmft = CDmft(archive = fname)
    analysis = ConvergenceAnalysis(cdmft, function)
    analysis.calc_standard_deviations([1025,1030])
    y = analysis.integrated_standard_deviations
    x = range(len(y))
    fig = plt.figure()
    ax1 = fig.add_axes([.15,.15,.82,.4])
    ax2 = fig.add_axes([.15,.55,.82,.4])
    n_colors = len(y[0]) * y[0].values()[0].shape[0]**2
    colors = [matplotlib.cm.jet(float(i)/max(1, n_colors - 1)) for i in range(n_colors)]
    c = 0
    for blockname, block in y[0].items():
        for i, j in itt.product(*[range(block.shape[0])]*2):
            ax1.plot(x, [yy[blockname][i, j].real for yy in y], label = blockname+str(i)+str(j), color = colors[c])
            ax2.plot(x, [yy[blockname][i, j].imag for yy in y], label = blockname+str(i)+str(j), color = colors[c])
            c += 1
    ax1.set_xlabel("$\mathrm{loop}$")
    ax1.set_ylabel("$\Re\Delta f$")
    ax2.set_ylabel("$\Im\Delta f$")
    ax2.set_xticklabels([])
    plt.savefig(fname[:-3]+"_conv_"+function+".pdf", dpi = 300)
    plt.close()
