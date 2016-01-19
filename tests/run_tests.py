from ttransformations import TestTransformations

import unittest

suite = unittest.TestSuite()
suite.addTest(TestTransformations('runDimerTransformation'))
suite.addTest(TestTransformations('runNambuTransformation'))
print
unittest.TextTestRunner(verbosity=2).run(suite)
