#!/usr/bin/env python3
import os
import unittest

from AaronTools.geometry import Geometry
from AaronTools.pathway import Pathway
from AaronTools.test import TestWithTimer, prefix, rmsd_tol
from numpy import array


class TestPathway(TestWithTimer):
    t60 = Geometry(os.path.join(prefix, "test_files", "torsion-60.xyz"))
    t90 = Geometry(os.path.join(prefix, "test_files", "torsion-90.xyz"))

    def test_interpolating_structure(self):
        # test to see if interpolated geometry is correct
        ref = Geometry(
            os.path.join(prefix, "ref_files", "torsion_interpolation.xyz")
        )
        pathway = Pathway(self.t60, array([self.t60.coords, self.t90.coords]))
        geom = pathway.geom_func(0.4)
        rmsd = geom.RMSD(ref, align=True, sort=False)
        self.assertTrue(rmsd < rmsd_tol(ref, superLoose=True))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestPathway("test_interpolating_structure"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
