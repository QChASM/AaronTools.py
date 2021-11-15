#!/usr/bin/env python3
import os
import unittest

import numpy as np
from AaronTools.fileIO import FileReader
from AaronTools.geometry import Geometry
from AaronTools.test import TestWithTimer, prefix


class TestOrbitals(TestWithTimer):
    small_mol = os.path.join(prefix, "test_files", "formaldehyde.fchk")

    def test_fchk_mo(self):
        ref = np.array([
            [ 1.98430354e-03, 1.14245330e+01, -2.01456705e-04, -2.01441878e-04],
            [ 7.31370333e+00, -1.94029767e-03, -1.36633289e-03, -1.36633695e-03],
            [ 0.9581908, 2.56044193, -0.03363992, -0.03363798],
            [ 1.33606002, -1.13133406, -0.24243181, -0.24241961],
            [ 2.36431044e-05, -2.45960625e-05, 2.49700776e-01, -2.49702951e-01],
            [ 0.26057303, -1.10826257, 0.10474883, 0.10475366],
            [ 2.41646200e-05, 1.99531504e-05, -2.00620098e-06, -2.00371274e-06],
            [-2.08098291e-06, 4.79409908e-07, 2.33169314e-01, -2.33170704e-01],
            [ 6.12950700e-05, -1.35701471e-05, 1.58944944e-05, 1.58759698e-05],
            [ 0.88931739, -0.05547194, 0.08981646, 0.08982103],
        ])
        
        fr = FileReader(TestOrbitals.small_mol, just_geom=False)
        geom = Geometry(fr)
        orbits = fr.other["orbitals"]
        coords = geom.coords
        for mo in range(0, 10):
            vals = orbits.mo_value(mo, coords)
            self.assertTrue(np.linalg.norm(vals - ref[mo]) < 1e-6)

    def test_fchk_density(self):
        ref = np.array([112.52268929, 279.16796565, 0.37519492, 0.37518832])
        fr = FileReader(TestOrbitals.small_mol, just_geom=False)
        geom = Geometry(fr)
        orbits = fr.other["orbitals"]
        coords = geom.coords
        vals = orbits.density_value(coords)
        self.assertTrue(np.linalg.norm(vals - ref) < 1e-6)

    def test_fchk_fukui_dual(self):
        ref = np.array([ 0.03395042, -0.00139618, -0.10751031, -0.10751156])
        fr = FileReader(TestOrbitals.small_mol, just_geom=False)
        geom = Geometry(fr)
        orbits = fr.other["orbitals"]
        coords = geom.coords
        vals = orbits.fukui_dual_value(coords)
        self.assertTrue(np.linalg.norm(vals - ref) < 1e-6)


if __name__ == "__main__":
    unittest.main()
