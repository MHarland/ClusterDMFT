#!/usr/bin/env pytriqs
import matplotlib, sys, itertools as itt, numpy as np
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from ClusterDMFT.convergence import ConvergenceAnalysis
from ClusterDMFT.cdmft import CDmft

arg_range = [1025, 1045]
first_loop_nr = 0

book = PdfPages("convergence.pdf")
for fname in sys.argv[1:]:
    cdmft = CDmft(archive = fname)
    fig = plt.figure()
    ax1 = fig.add_axes([.12,.1,.76,.85/4])
    ax2 = fig.add_axes([.12,.1+.85/4,.76,.85/4])
    ax3 = fig.add_axes([.12,.1+2*.85/4,.76,.85/4])
    ax4 = fig.add_axes([.12,.1+3*.85/4,.76,.85/4])
    axes = [ax1, ax2, ax3, ax4]
    x = range(first_loop_nr, cdmft.next_loop())
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
                y = []
                for loopnr in range(first_loop_nr, cdmft.next_loop()):
                    s = cdmft.load('sigma_transf_iw', loopnr)
                    y.append(s[blockname].data[n_iw + freq, i, j])
                    rey = np.array(y).real
                    imy = np.array(y).imag
                    rey -= rey[0]
                    imy -= imy[0]
                ax1.plot(x, rey, label = blockname+str(i)+str(j), color = colors[c])
                ax2.plot(x, imy, label = blockname+str(i)+str(j), color = colors[c])
            c += 1
    y = []
    y2 = []
    ax1b = ax1.twinx()
    for loopnr in range(first_loop_nr, cdmft.next_loop()):
        dmu = cdmft.load('dmu', loopnr)
        density = cdmft.load('density', loopnr)
        y.append(dmu)
        y2.append(density)
    y = np.array(y) - y[0]
    y2 = np.array(y2) - y2[0]
    ax1b.plot(x, y, label = "$\\tilde{mu}$", color = 'black')
    ax1b.plot(x, y2, label = "$N$", color = 'gray')
    ax1b.set_ylabel("$\\tilde{\mu}$")
    ax2.set_ylabel("$\Im\\tilde{\Sigma}(i\\omega_0)$")
    ax1.set_ylabel("$\Re\\tilde{\Sigma}(i\\omega_0)$")
    ax2.set_xticklabels([])
    ax1.set_xlabel("$\\mathrm{loop}$")    
    
    c = 0
    for blocknr, blockname in enumerate(blocknames):
        for i, j in itt.product(*[blockindices[blocknr]]*2):
        #for i, j in itt.product(*[blockindices]*2):
            for freq in [0]:
                y = []
                for loopnr in range(first_loop_nr, cdmft.next_loop()):
                    s = cdmft.load('g_transf_iw', loopnr)
                    y.append(s[blockname].data[n_iw + freq, i, j])
                    rey = np.array(y).real
                    imy = np.array(y).imag
                    rey -= rey[0]
                    imy -= imy[0]
                ax3.plot(x, rey, label = blockname+str(i)+str(j), color = colors[c])
                ax4.plot(x, imy, label = blockname+str(i)+str(j), color = colors[c])
            c += 1
    y = []
    ax4.set_ylabel("$\Im\\tilde{G}(i\\omega_0)$")
    ax3.set_ylabel("$\Re\\tilde{G}(i\\omega_0)$")
    ax3.set_xticklabels([])
    ax4.set_xticklabels([])
    fig.suptitle(fname)
    #plt.savefig(fname[:-3]+"_conv.pdf", dpi = 300)
    book.savefig()
    plt.close()
    
book.close()
