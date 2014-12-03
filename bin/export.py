#!/usr/bin/env pytriqs
from cdmft.cellular_dmft import CDmft
import sys

for arg in sys.argv[1:len(sys.argv)]:
    print 'exporting ' + arg + '...'
    x = CDmft(archive = arg)
    x.export_results()
    del x
    print arg[0:-3] + '.pdf ready'
