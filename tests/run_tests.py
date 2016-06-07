from ttransformations import TestTransformations
from tconvergence import TestConvergence

import unittest

suite = unittest.TestSuite()
suite.addTest(TestTransformations('runDimerTransformation'))
suite.addTest(TestTransformations('runNambuTransformation'))
suite.addTest(TestConvergence('run_StandardDeviation'))
suite.addTest(TestConvergence('run_Distance'))
suite.addTest(TestConvergence('run_ConvergenceAnalysis'))
print
unittest.TextTestRunner(verbosity=2).run(suite)
