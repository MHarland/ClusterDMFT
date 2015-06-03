#!/usr/bin/env pytriqs
from cdmft.cdmft import CDmft
import sys

for arg in sys.argv[1:len(sys.argv)]:
    print 'exporting ' + arg + '...'
    x = CDmft(archive = arg)
    x.export_results()
    del x
