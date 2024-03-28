## This file contains helpers that can be used across unit tests.

import sys
import os
import numpy as np

sourceDirPath = "../src"
sqrtSymbol = "\u221A"

# Add src directory to path
def addSourceDirectoryToPath():
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), sourceDirPath))
    sys.path.insert(0, src_dir)

def compareMatricies(testObject, a, b, places=0):
    if not (isinstance(a, np.ndarray) or isinstance(b, np.ndarray)):
        if places == 0:
            testObject.assertEqual(a,b)
        else:
            testObject.assertAlmostEqual(a,b,places=places)
    elif(a.shape != b.shape):
        testObject.fail("Shapes do not match. " + str(a.shape) + " != " + str(b.shape))
    else:
        for row in range(a.shape[0]):
            compareMatricies(testObject, a[row],b[row], places=places)