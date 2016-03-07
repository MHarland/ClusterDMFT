#!/usr/bin/env pytriqs
from ClusterDMFT.cdmft import CDmft
import sys

for arch in sys.argv[1:]:
    results = CDmft(archive = arch)
    sign = results.load('sign')
    print arch[:-3] + ':'
    print 'sign: '+str(sign)
    print
