#!/usr/bin/env pytriqs
from ClusterDMFT.cdmft import CDmft
from pytriqs.plot.mpl_interface import oplot
import sys
from matplotlib import pyplot as plt

max_freq = int(sys.argv[1])

for n, arch in enumerate(sys.argv[2:]):
    c = CDmft(archive = arch)
    g = c.load('g_c_iw')
    for s, b in g:
        for i in range(len(b.data[0,:,:])):
            for j in range(len(b.data[0,:,:])):
                oplot(b[i, j], x_window = (0, max_freq), RI = 'R', name = 'Re')
                oplot(b[i, j], x_window = (0, max_freq), RI = 'I', name = 'Im')
                if s == 'up': sout = '\\uparrow'
                if s == 'down': sout = '\\downarrow'
                plt.gca().set_ylabel('$G_{'+sout+'\\,'+str(i)+str(j)+'}^{(c)}(i\\omega_n)$')
                plt.savefig(arch[0:-3]+'_G_'+s+'_'+str(i)+str(j)+'.png', dpi = 300)
                plt.close()
