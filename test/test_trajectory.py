#!/usr/bin/env python3
import os
import unittest

from numpy import dot, finfo
from numpy.linalg import inv

from AaronTools.geometry import Geometry
from AaronTools.test import TestWithTimer, prefix, rmsd_tol
from AaronTools.trajectory import Pathway


class TestPathway(TestWithTimer):
    t60 = Geometry(os.path.join(prefix, "test_files/torsion-60.xyz"))
    t90 = Geometry(os.path.join(prefix, "test_files/torsion-90.xyz"))

    def test_interpolating_structure(self):
        # test to see if interpolated geometry is correct
        ref = Geometry(
            os.path.join(prefix, "ref_files/torsion_interpolation.xyz")
        )
        S = Pathway([self.t60, self.t90])
        geom = S.Geom_func(0.4)
        rmsd = geom.RMSD(ref, align=True)
        self.assertTrue(rmsd < rmsd_tol(ref, superLoose=True))

    def test_splines_values(self):
        # test cubic splines function values
        # ought to have two splines:
        #   g(x) = -10x^3 +  15x^2
        #   h(x) =  10x^3 + -15x^2  + 5
        ref = [0, 0.78125, 2.5, 5, 4.21875, 2.5, 0]
        ref_d = [0, 5.625, 7.5, 0, -5.625, -7.5, 0]
        test_t = [0, 0.125, 0.25, 0.5, 0.625, 0.75, 1]
        tolerance = 50 * finfo(float).eps
        ev = [0, 5, 0]
        m = Pathway.get_splines_mat(3)
        mi = inv(m)
        b = Pathway.get_splines_vector(ev)
        c = dot(mi, b)
        f, df = Pathway.get_E_func(c, [1, 1])
        for i in range(0, len(test_t)):
            v = f(test_t[i])
            dv = df(test_t[i])
            self.assertTrue(abs(v - ref[i]) <= tolerance)
            self.assertTrue(abs(dv - ref_d[i]) <= tolerance)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestPathway("test_interpolating_structure"))
    suite.addTest(TestPathway("test_splines_values"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
