import time
import unittest
import numpy as np

from os.path import dirname

prefix = dirname(__file__)
if prefix:
    prefix += '/'

def rmsd_tol(geom, superTight=False, superLoose=False):
    """automatically determine a reasonable rmsd tolerance for the input
    geometry based on its size and number of atoms"""
    tolerance = len(geom.atoms)**(2-int(superTight)+int(superLoose)) * np.sqrt(np.finfo(float).eps)
    
    com = geom.COM()
    max_d = None
    for atom in geom.atoms:
        d = np.linalg.norm(atom.coords-com)
        if max_d is None or d > max_d:
            max_d = d

    tolerance *= max_d*(2+int(superLoose))

    tolerance = tolerance**(2/(2-int(superTight)+int(superLoose)))

    return tolerance

class TestWithTimer(unittest.TestCase):
    count = 0
    total_time = 0
    last_class = None
    last_result = None
    _errors = 0
    _fails = 0

    @classmethod
    def tearDownClass(cls):
        errors = len(cls.last_result.errors)
        fails = len(cls.last_result.failures)
        if TestWithTimer._errors != errors:
            TestWithTimer._errors = errors
            print('\b ERROR')
        elif TestWithTimer._fails != fails:
            TestWithTimer._fails = fails
            print('\b FAIL')
        else:
            print('\b ok')

        print("Ran {} tests in {:.3f}s".format(
            TestWithTimer.count, TestWithTimer.total_time))
        print("-"*70)

    def setUp(self):
        errors = len(self._outcome.result.errors)
        fails = len(self._outcome.result.failures)
        if TestWithTimer._errors != errors:
            TestWithTimer._errors = errors
            print('\b ERROR')
        elif TestWithTimer._fails != fails:
            TestWithTimer._fails = fails
            print('\b FAIL')
        elif self.id().split('.')[1] == TestWithTimer.last_class:
            print('\b ok')
        elif TestWithTimer.last_class is not None:
            TestWithTimer.total_time = 0
        self.start_time = time.time()

    def tearDown(self):
        t = time.time() - self.start_time
        TestWithTimer.total_time += t

        name = self.id().split('.')[1:]
        if not TestWithTimer.last_class or TestWithTimer.last_class != name[0]:
            TestWithTimer.last_class = name[0]
            TestWithTimer.count = 0
            print("\r{}:".format(name[0]))

        name = name[1]
        TestWithTimer.count += 1
        print("\r    {:3.0f}: {:<30s} {:.3f}s  ".format(
            TestWithTimer.count, name, t), end='')

        TestWithTimer.last_result = self._outcome.result
