#!/usr/bin/env python3
import os
import unittest

from AaronTools.geometry import Geometry
from AaronTools.test import TestWithTimer, prefix, rmsd_tol
from AaronTools.finders import BondsFrom


class TestPathway(TestWithTimer):
    benzene = os.path.join(prefix, "test_files", "benzene.xyz")

    def test_bonds_finder(self):
        # test to see if interpolated geometry is correct
        mol = Geometry(self.benzene)
        
        h1 = mol.find('H')

        para_finder = BondsFrom(h1, 5)
        out = mol.find(para_finder, 'H')
        self.assertTrue(len(out) == 1)

        meta_finder = BondsFrom(h1, 4)
        out = mol.find(meta_finder, 'H')
        self.assertTrue(len(out) == 2)

        ortho_finder = BondsFrom(h1, 3)
        out = mol.find(ortho_finder, 'H')
        self.assertTrue(len(out) == 2)




def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestPathway("test_bonds_finder"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
