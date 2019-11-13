#!/usr/bin/env python3
import os
import unittest

from AaronTools.component import Component
from AaronTools.substituent import Substituent
from AaronTools.test import TestWithTimer, prefix, rmsd_tol


class TestComponent(TestWithTimer):
    # simple geometries
    benz = Component(os.path.join(prefix, "test_files/benzene.xyz"))
    benz_Cl = Component(os.path.join(prefix, "test_files/benzene_4-Cl.xyz"))
    benz_NO2_Cl = Component(
        os.path.join(prefix, "test_files/benzene_1-NO2_4-Cl.xyz")
    )
    benz_OH_Cl = Component(
        os.path.join(prefix, "test_files/benzene_1-OH_4-Cl.xyz")
    )
    benz_Ph_Cl = Component(
        os.path.join(prefix, "test_files/benzene_1-Ph_4-Cl.xyz")
    )
    Et_NO2 = Component(os.path.join(prefix, "test_files/Et_1-NO2.xyz"))
    pent = Component(os.path.join(prefix, "test_files/pentane.xyz"))

    # ligands
    RQ_tBu = Component(os.path.join(prefix, "test_files/R-Quinox-tBu3.xyz"))
    for a in RQ_tBu.find("P"):
        a.add_tag("key")
    tri = Component(os.path.join(prefix, "test_files/ligands/squaramide.xyz"))

    def is_member(self, valid, test):
        for a in valid:
            try:
                test.remove(a)
            except KeyError:
                return False
        if len(test) == 0:
            return True
        else:
            return False

    def is_same(self, valid, test):
        # same number of atoms
        if len(valid.atoms) != len(test.atoms):
            return False
        # of same elements
        tmp = [a.element for a in test.atoms]
        for e in [a.element for a in valid.atoms]:
            try:
                tmp.remove(e)
            except ValueError:
                return False
        # with reasonable rmsd
        if valid.RMSD(test, sort=True) > 10 ** -4:
            return False
        return True

    def test_substitute(self):
        mol = TestComponent.benz.copy()
        benz_Cl = TestComponent.benz_Cl.copy()
        benz_NO2_Cl = TestComponent.benz_NO2_Cl.copy()
        benz_OH_Cl = TestComponent.benz_OH_Cl.copy()
        benz_Ph_Cl = TestComponent.benz_Ph_Cl.copy()

        mol.substitute(Substituent("Cl"), "11")
        rmsd = mol.RMSD(benz_Cl, sort=True)
        self.assertTrue(rmsd < rmsd_tol(benz_Cl))

        mol.substitute(Substituent("NO2"), "12", "1")
        rmsd = mol.RMSD(benz_NO2_Cl, sort=True)
        self.assertTrue(rmsd < rmsd_tol(benz_NO2_Cl))

        mol.substitute(Substituent("OH"), "NO2")
        rmsd = mol.RMSD(benz_OH_Cl, sort=True)
        self.assertTrue(rmsd < rmsd_tol(benz_OH_Cl))

        mol.substitute(Substituent("Ph"), "12.*")
        rmsd = mol.RMSD(benz_Ph_Cl)
        self.assertTrue(rmsd < rmsd_tol(benz_Ph_Cl))
        
    def test_detect_backbone(self):
        geom = TestComponent.RQ_tBu.copy()
        geom.detect_backbone()

        backbone = geom.find("1,2,7-20")
        Me = geom.find("3,21-23")
        tBu = geom.find("4-6,24-59")

        try:
            test_Me = set(geom.find("Me"))
            test_tBu = set(geom.find("tBu"))
            test_backbone = set(geom.find("backbone"))
        except LookupError:
            self.assertTrue(False)

        self.assertTrue(self.is_member(Me, test_Me))
        self.assertTrue(self.is_member(tBu, test_tBu))
        self.assertTrue(self.is_member(backbone, test_backbone))

    def test_minimize_torsion(self):
        geom = TestComponent.benz.copy()
        ref = Component("ref_files/minimize_torsion.xyz")
        
        geom.substitute(Substituent("tBu"), "12")
        geom.substitute(Substituent("Ph"), "10")
        geom.substitute(Substituent("OH"), "7")

        geom.minimize_sub_torsion()
        rmsd = geom.RMSD(ref, align=True)
        self.assertTrue(rmsd < 1)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestComponent("test_detect_backbone"))
    suite.addTest(TestComponent("test_substitute"))
    suite.addTest(TestComponent("test_minimize_torsion"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
