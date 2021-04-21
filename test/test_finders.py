#!/usr/bin/env python3
import os
import unittest

from AaronTools.geometry import Geometry
from AaronTools.test import TestWithTimer, prefix, rmsd_tol
from AaronTools.finders import *


class TestFinder(TestWithTimer):
    benzene = os.path.join(prefix, "test_files", "benzene.xyz")
    catalyst = os.path.join(prefix, "test_files", "catalysts", "tm_single-lig.xyz")
    ligand = os.path.join(prefix, "test_files", "R-Quinox-tBu3.xyz")

    chiral_ring = os.path.join(prefix, "test_files", "chiral_ring.xyz")
    chiral_mol_1 = os.path.join(prefix, "test_files", "chiral_centers_1.xyz")
    chiral_mol_2 = os.path.join(prefix, "test_files", "chiral_centers_2.xyz")
    chiral_mol_3 = os.path.join(prefix, "test_files", "chiral_centers_3.xyz")
    chiral_mol_4 = os.path.join(prefix, "test_files", "chiral_centers_4.xyz")

    def test_BondsFrom(self):
        # test to see if interpolated geometry is correct
        mol = Geometry(self.benzene)
        
        h1 = mol.find('H')[0]

        para_finder = BondsFrom(h1, 5)
        out = mol.find(para_finder, 'H')
        self.assertTrue(all([atom in mol.find('9') for atom in out]))

        meta_finder = BondsFrom(h1, 4)
        out = mol.find(meta_finder, 'H')
        self.assertTrue(all([atom in mol.find('10,11') for atom in out]))

        ortho_finder = BondsFrom(h1, 3)
        out = mol.find(ortho_finder, 'H')
        self.assertTrue(all([atom in mol.find('8,12') for atom in out]))

    def test_WithinBondsOf(self):
        mol = Geometry(self.benzene)

        h1 = mol.find('H')[0]

        out = mol.find(WithinBondsOf(h1, 2))
        self.assertTrue(all([atom in mol.find('1,2,3') for atom in out]))

    def test_BondedTo(self):
        mol = Geometry(self.benzene)

        c1 = mol.find('C')[0]

        out = mol.find(BondedTo(c1))
        self.assertTrue(all([atom in mol.find('2,6,12') for atom in out]))

    def test_WithinRadiusFromPoint(self):
        mol = Geometry(self.benzene)
        mol.coord_shift(-mol.COM())

        out = mol.find(WithinRadiusFromPoint([0, 0, 0], 1.5))
        self.assertTrue(all([atom in mol.find('C') for atom in out]))

    def test_WithinRadiusFromAtom(self):
        mol = Geometry(self.benzene)
        
        c1 = mol.find('C')[0]

        out = mol.find(WithinRadiusFromAtom(c1, 1.5))
        self.assertTrue(all([atom in mol.find('1,2,6,12') for atom in out]))

    def test_NotAny(self):
        mol = Geometry(self.benzene)

        out = mol.find(NotAny('C', '12'))
        self.assertTrue(all([atom in mol.find('7,8,9,10,11') for atom in out]))

    def test_AnyTransitionMetal(self):
        mol = Geometry(self.catalyst)

        out = mol.find(AnyTransitionMetal())
        self.assertTrue(all([atom in mol.find('Cu') for atom in out]))

    def test_AnyNonTransitionMetal(self):
        mol = Geometry(self.benzene)

        out = mol.find(AnyNonTransitionMetal())
        self.assertTrue(all([atom in mol.find('C,H') for atom in out]))

    def test_VSEPR(self):
        mol = Geometry(self.benzene)

        out = mol.find(VSEPR('trigonal planar'))
        self.assertTrue(all([atom in mol.find('C') for atom in out]))

    def test_BondedElements(self):
        mol = Geometry(self.benzene)

        out = mol.find(BondedElements('C', 'C', 'H'))
        self.assertTrue(all([atom in mol.find('C') for atom in out]))

    def test_NumberOfBonds(self):
        mol = Geometry(self.benzene)

        out = mol.find(NumberOfBonds(1))
        self.assertTrue(all([atom in mol.find('H') for atom in out]))

        out = mol.find(NumberOfBonds(3))
        self.assertTrue(all([atom in mol.find('C') for atom in out]))
    
    def test_ChiralCenters(self):
        mol = Geometry(self.ligand)

        out = mol.find(ChiralCenters())
        self.assertTrue(all([atom in mol.find('1') for atom in out]))


        mol = Geometry(self.chiral_ring)
        out = mol.find(ChiralCenters())
        self.assertTrue(all([atom in mol.find('3,12') for atom in out]))

        #next two tests are finding chiral centers that are chiral
        #b/c of other chiral centers
        mol = Geometry(self.chiral_mol_1)
        out = mol.find(ChiralCenters())
        self.assertTrue(all([atom in mol.find('1,3,5') for atom in out]))

        mol = Geometry(self.chiral_mol_2)
        out = mol.find(ChiralCenters())
        self.assertTrue(all([atom in mol.find('1,4') for atom in out]))

        #chiral centers in rings
        mol = Geometry(self.chiral_mol_3)
        out = mol.find(ChiralCenters())
        self.assertTrue(all([atom in mol.find('17,18') for atom in out]))

        mol = Geometry(self.chiral_mol_4)
        out = mol.find(ChiralCenters())
        self.assertTrue(all([atom in mol.find('9,17,18') for atom in out]))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestFinder("test_BondsFrom"))
    suite.addTest(TestFinder("test_WithinBondsOf"))
    suite.addTest(TestFinder("test_BondedTo"))
    suite.addTest(TestFinder("test_WithinRadiusFromPoint"))
    suite.addTest(TestFinder("test_WithinRadiusFromAtom"))
    suite.addTest(TestFinder("test_NotAny"))
    suite.addTest(TestFinder("test_AnyTransitionMetal"))
    suite.addTest(TestFinder("test_AnyNonTransitionMetal"))
    suite.addTest(TestFinder("test_VSEPR"))
    suite.addTest(TestFinder("test_BondedElements"))
    suite.addTest(TestFinder("test_NumberOfBonds"))
    suite.addTest(TestFinder("test_ChiralCenters"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
