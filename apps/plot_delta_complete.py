#!/usr/bin/env pytriqs
from ClusterDMFT.cdmft import CDmft
from pytriqs.plot.mpl_interface import oplot
import sys
from matplotlib import pyplot as plt

for n, arch in enumerate(sys.argv[1:]):
    c = CDmft(archive = arch)
    g = c.load('Delta_sym_tau')
    for s, b in g:
        for i in range(len(b.data[0,:,:])):
            for j in range(len(b.data[0,:,:])):
                oplot(b[i, j])
                plt.gca().set_ylabel('$\Delta_{'+s+'\\,'+str(i)+str(j)+'}^{(s)}(\\tau)$')
                plt.savefig(arch[0:-3]+'_Delta_'+s+'_'+str(i)+str(j)+'.png', dpi = 300)
                plt.close()
