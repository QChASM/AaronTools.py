#!/usr/bin/env python3
import os
import unittest

from AaronTools.geometry import Geometry
from AaronTools.test import TestWithTimer, prefix, rmsd_tol
from AaronTools.symmetry import PointGroup

class TestFinder(TestWithTimer):
    benzene = os.path.join(prefix, "test_files", "benzene.xyz")
    c60 = os.path.join(prefix, "test_files", "c60.xyz")
    adamantane = os.path.join(prefix, "test_files", "adamantane.xyz")
    methane = os.path.join(prefix, "test_files", "methane.xyz")
    c3h4 = os.path.join(prefix, "test_files", "c3h4.xyz")
    h2o = os.path.join(prefix, "test_files", "h2o.xyz")
    meoh = os.path.join(prefix, "test_files", "MeOH.xyz")
    Me_NO2_4 = os.path.join(prefix, "test_files", "C(NO2)4.xyz")

    chiral_ring = os.path.join(prefix, "test_files", "chiral_ring.xyz")
    chiral_mol_1 = os.path.join(prefix, "test_files", "chiral_centers_1.xyz")
    chiral_mol_2 = os.path.join(prefix, "test_files", "chiral_centers_2.xyz")
    chiral_mol_3 = os.path.join(prefix, "test_files", "chiral_centers_3.xyz")
    chiral_mol_4 = os.path.join(prefix, "test_files", "chiral_centers_4.xyz")

    def test_D6h(self):
        # test to see if interpolated geometry is correct
        mol = Geometry(self.benzene, refresh_ranks=False)
        pg = PointGroup(mol)
        
        self.assertEqual(pg.name, "D6h")

    def test_Cs(self):
        # test to see if interpolated geometry is correct
        mol = Geometry(self.meoh, refresh_ranks=False)
        pg = PointGroup(mol)
        
        self.assertEqual(pg.name, "Cs")

    def test_S4(self):
        # test to see if interpolated geometry is correct
        mol = Geometry(self.Me_NO2_4, refresh_ranks=False)
        pg = PointGroup(mol)
        
        self.assertEqual(pg.name, "S4")

    def test_C1(self):
        mol1 = Geometry(self.chiral_ring, refresh_ranks=False)
        pg = PointGroup(mol1)

        self.assertEqual(pg.name, "C1")

        mol2 = Geometry(self.chiral_mol_1, refresh_ranks=False)
        pg = PointGroup(mol2)

        self.assertEqual(pg.name, "C1")

        mol3 = Geometry(self.chiral_mol_2, refresh_ranks=False)
        pg = PointGroup(mol3)

        self.assertEqual(pg.name, "C1")

        mol4 = Geometry(self.chiral_mol_3, refresh_ranks=False)
        pg = PointGroup(mol4)
        
        self.assertEqual(pg.name, "C1")

        mol5 = Geometry(self.chiral_mol_4, refresh_ranks=False)
        pg = PointGroup(mol5)
        
        self.assertEqual(pg.name, "C1")

    def test_Ih(self):
        mol = Geometry(self.c60, refresh_ranks=False)
        pg = PointGroup(mol)
        
        self.assertEqual(pg.name, "Ih")

    def test_Td(self):
        mol = Geometry(self.adamantane, refresh_ranks=False)
        pg = PointGroup(mol)
        
        self.assertEqual(pg.name, "Td")
        
        mol = Geometry(self.methane, refresh_ranks=False)
        pg = PointGroup(mol)
        
        self.assertEqual(pg.name, "Td")

    def test_D2d(self):
        mol = Geometry(self.c3h4, refresh_ranks=False)
        pg = PointGroup(mol)
        
        self.assertEqual(pg.name, "D2d")

    def test_C2v(self):
        mol = Geometry(self.h2o, refresh_ranks=False)
        pg = PointGroup(mol)
        
        self.assertEqual(pg.name, "C2v")


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestFinder("test_D6h"))
    suite.addTest(TestFinder("test_Cs"))
    suite.addTest(TestFinder("test_S4"))
    suite.addTest(TestFinder("test_C1"))
    suite.addTest(TestFinder("test_Ih"))
    suite.addTest(TestFinder("test_Td"))
    suite.addTest(TestFinder("test_D2d"))
    suite.addTest(TestFinder("test_C2v"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
