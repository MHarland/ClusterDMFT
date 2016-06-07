#!/usr/bin/env pytriqs
import matplotlib, sys, itertools as itt, numpy as np
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from ClusterDMFT.convergence import ConvergenceAnalysis
from ClusterDMFT.cdmft import CDmft

tau_range = [4975, 5025] #[4900, 5100]
first_loop_nr = int(sys.argv[1])

for fname in sys.argv[2:]:
    cdmft = CDmft(archive = fname)
    fig = plt.figure()
    ax1 = fig.add_axes([.11,.1,.77,.87/4])
    ax2 = fig.add_axes([.11,.1+.87/4,.77,.87/4])
    ax3 = fig.add_axes([.11,.1+2*.87/4,.77,.87/4])
    ax4 = fig.add_axes([.11,.1+3*.87/4,.77,.87/4])

    analysis = ConvergenceAnalysis(cdmft, "g_transf_tau", first_loop_nr)
    analysis.calc_standard_deviations(tau_range)
    y = analysis.integrated_standard_deviations[1:]
    n_colors = len(y[0]) * y[0].values()[0].shape[0]**2
    colors = [matplotlib.cm.jet(float(i)/max(1, n_colors - 1)) for i in range(n_colors)]
    x = range(first_loop_nr, cdmft.next_loop())
    c = 0
    for blockname, block in y[0].items():
        for i, j in itt.product(*[range(block.shape[0])]*2):
            ax1.semilogy(x[::-1], [yy[blockname][i, j].real for yy in y], label = blockname+str(i)+str(j), color = colors[c])
            c += 1
    ax1.set_xlabel("$\mathrm{loop}$")
    ax1.set_ylabel("$\sigma \\tilde{G}(\\tau)$")

    analysis.calc_distances(tau_range)
    y = analysis.integrated_distances
    #x = range(len(y))
    c = 0
    for blockname, block in y[0].items():
        for i, j in itt.product(*[range(block.shape[0])]*2):
            ax2.semilogy(x, [abs(yy[blockname][i, j].real) for yy in y], label = blockname+str(i)+str(j), color = colors[c])
            c += 1
    ax2.set_ylabel("$\Delta \\tilde{G}(\\tau)$")
    ax2.set_xticklabels([])

    p = cdmft.load('parameters')
    n_iw = p['n_iw']
    g_struct = p['g_transf_struct']
    blocknames = [block[0] for block in g_struct]
    blockindices = [block[1] for block in g_struct]
    #blocknames = p['blocks']
    #blockindices = p['blockstates']
    #n_colors = len(blocknames) * len(blockindices)**2
    n_colors = len(blocknames) * len(blockindices[0])**2
    
    colors = [matplotlib.cm.jet(float(i)/max(1, n_colors - 1)) for i in range(n_colors)]
    c = 0
    for blocknr, blockname in enumerate(blocknames):
        for i, j in itt.product(*[blockindices[blocknr]]*2):
        #for i, j in itt.product(*[blockindices]*2):
            for freq in [0]:
                #x = []
                y = []
                for loopnr in range(first_loop_nr, cdmft.next_loop()):
                    s = cdmft.load('g_transf_iw', loopnr)
                    x.append(loopnr)
                    y.append(s[blockname].data[n_iw + freq, i, j])
                ax4.plot(x, np.array(y).real, label = blockname+str(i)+str(j), color = colors[c])
                ax3.plot(x, np.array(y).imag, label = blockname+str(i)+str(j), color = colors[c])
            c += 1
    #x = []
    y = []
    ax4b = ax4.twinx()
    for loopnr in range(first_loop_nr, cdmft.next_loop()):
        dmu = cdmft.load('dmu', loopnr)
        x.append(loopnr)
        y.append(dmu)
    ax4b.plot(x, y, label = "$\\tilde{mu}$", color = 'gray')
    ax4b.set_ylabel("$\\tilde{\mu}$")
    ax3.set_ylabel("$\Im\\tilde{\Sigma}(i\\omega_0)$")
    ax4.set_ylabel("$\Re\\tilde{\Sigma}(i\\omega_0)$")
    ax3.set_xticklabels([])
    ax4.set_xticklabels([])
    
    plt.savefig(fname[:-3]+"_conv.pdf", dpi = 300)
    plt.close()
