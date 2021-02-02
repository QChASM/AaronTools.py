#!/usr/bin/env python3
import os
import unittest
import numpy as np

from AaronTools.component import Component
from AaronTools.geometry import Geometry
from AaronTools.substituent import Substituent
from AaronTools.test import TestWithTimer, prefix, validate


class TestComponent(TestWithTimer):
    # simple geometries
    benz = Component(os.path.join(prefix, "test_files", "benzene.xyz"))
    benz_Cl = Component(os.path.join(prefix, "test_files", "benzene_4-Cl.xyz"))
    benz_NO2_Cl = Component(
        os.path.join(prefix, "test_files", "benzene_1-NO2_4-Cl.xyz")
    )
    benz_OH_Cl = Component(
        os.path.join(prefix, "test_files", "benzene_1-OH_4-Cl.xyz")
    )
    benz_Ph_Cl = Component(
        os.path.join(prefix, "test_files", "benzene_1-Ph_4-Cl.xyz")
    )
    Et_NO2 = Component(os.path.join(prefix, "test_files", "Et_1-NO2.xyz"))
    pent = Component(os.path.join(prefix, "test_files", "pentane.xyz"))

    # ligands
    RQ_tBu = Component(os.path.join(prefix, "test_files", "R-Quinox-tBu3.xyz"))
    for a in RQ_tBu.find("P"):
        a.add_tag("key")
    tri = Component(
        os.path.join(prefix, "test_files", "ligands", "squaramide.xyz")
    )

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

    def test_substitute(self):
        mol = TestComponent.benz.copy()
        benz_Cl = TestComponent.benz_Cl.copy()
        benz_NO2_Cl = TestComponent.benz_NO2_Cl.copy()
        benz_OH_Cl = TestComponent.benz_OH_Cl.copy()
        benz_Ph_Cl = TestComponent.benz_Ph_Cl.copy()

        mol.substitute(Substituent("Cl"), "11")
        res = validate(mol, benz_Cl)
        self.assertTrue(res)

        mol.substitute(Substituent("NO2"), "12", "1")
        res = validate(mol, benz_NO2_Cl, sort=True)
        self.assertTrue(res)

        mol.substitute(Substituent("OH"), "NO2")
        res = validate(mol, benz_OH_Cl, sort=True)
        self.assertTrue(res)

        mol.substitute(Substituent("Ph"), ["12", "12.*"])
        res = validate(mol, benz_Ph_Cl, sort=True, thresh="loose")
        self.assertTrue(res)

    def test_detect_backbone(self):
        geom = TestComponent.RQ_tBu.copy()

        backbone = geom.find("1,2,7-20")
        Me = geom.find("3,21-23")
        tBu = geom.find("4-6,24-59")

        try:
            test_Me = set(geom.find(["Me", "CH3"]))
            test_tBu = set(geom.find("tBu"))
            test_backbone = set(geom.find("backbone"))
        except LookupError:
            self.assertTrue(False)

        self.assertTrue(self.is_member(Me, test_Me))
        self.assertTrue(self.is_member(tBu, test_tBu))
        self.assertTrue(self.is_member(backbone, test_backbone))

    def test_minimize_torsion(self):
        ref = Component(
            os.path.join(prefix, "ref_files", "minimize_torsion.xyz")
        )

        geom = TestComponent.benz.copy()
        geom.substitute(Substituent("tBu"), "12")
        geom.substitute(Substituent("Ph"), "10")
        geom.substitute(Substituent("OH"), "7")
        geom.minimize_sub_torsion()
        self.assertTrue(validate(geom, ref, heavy_only=True))

    def test_sub_rotate(self):
        geom = TestComponent.RQ_tBu.copy()
        geom.sub_rotate("4", angle=60)
        geom.sub_rotate("6", angle=60)
        geom.sub_rotate("5", angle=60)
        ref = Component(os.path.join(prefix, "ref_files", "sub_rotate.xyz"))
        self.assertTrue(validate(geom, ref, heavy_only=True))


ONLYSOME = False


def suite():
    suite = unittest.TestSuite()
    # suite.addTest(TestComponent("test_detect_backbone"))
    # suite.addTest(TestComponent("test_sub_rotate"))
    # suite.addTest(TestComponent("test_substitute"))
    # suite.addTest(TestComponent("test_minimize_torsion"))
    suite.addTest(TestComponent("test_cone_angle"))
    return suite


if __name__ == "__main__" and ONLYSOME:
    runner = unittest.TextTestRunner()
    runner.run(suite())
elif __name__ == "__main__":
    unittest.main()
