#!/usr/bin/env pytriqs
from cdmft.cdmft import CDmft
import sys
from matplotlib import pyplot as plt, cm

for n, arch in enumerate(sys.argv[1:]):
    c = CDmft(archive = arch)
    s = c.load('spectrum')
    plt.scatter(range(len(s)), s, label = arch[0:-3], color = cm.jet(n /float(len(sys.argv[1:]))))

plt.legend(loc = 2)
plt.savefig('spectrum.pdf', dpi = 300)
