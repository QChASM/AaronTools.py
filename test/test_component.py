#!/usr/bin/env python3
import unittest

from AaronTools.component import Component
from AaronTools.substituent import Substituent
from AaronTools.test import prefix, TestWithTimer


class TestComponent(TestWithTimer):
    # simple geometries
    benz = Component(prefix + "test_files/benzene.xyz")
    benz_Cl = Component(prefix + "test_files/benzene_4-Cl.xyz")
    benz_NO2_Cl = Component(prefix + "test_files/benzene_1-NO2_4-Cl.xyz")
    benz_OH_Cl = Component(prefix + "test_files/benzene_1-OH_4-Cl.xyz")
    benz_Ph_Cl = Component(prefix + "test_files/benzene_1-Ph_4-Cl.xyz")
    Et_NO2 = Component(prefix + "test_files/Et_1-NO2.xyz")
    pent = Component(prefix + "test_files/pentane.xyz")

    # ligands
    RQ_tBu = Component(prefix + "test_files/R-Quinox-tBu3.xyz")
    for a in RQ_tBu.find('P'):
        a.add_tag('key')
    tri = Component(prefix + "test_files/ligands/squaramide.xyz")

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
        if valid.RMSD(test, sort=True) > 10**-4:
            return False
        return True

    def test_substitute(self):
        mol = TestComponent.benz.copy()
        benz_Cl = TestComponent.benz_Cl.copy()
        benz_NO2_Cl = TestComponent.benz_NO2_Cl.copy()
        benz_OH_Cl = TestComponent.benz_OH_Cl.copy()
        benz_Ph_Cl = TestComponent.benz_Ph_Cl.copy()

        mol.substitute(Substituent('Cl'), '11')
        rmsd = mol.RMSD(benz_Cl)
        self.assertTrue(rmsd < 10**-4)

        mol.substitute(Substituent('NO2'), '12', '1')
        rmsd = mol.RMSD(benz_NO2_Cl)
        mol.write('tmp')
        self.assertTrue(rmsd < 10**-4)

        mol.substitute('OH', 'NO2')
        rmsd = mol.RMSD(benz_OH_Cl)
        self.assertTrue(rmsd < 10**-4)

        mol.substitute('Ph', '12.*')
        rmsd = mol.RMSD(benz_Ph_Cl)
        self.assertTrue(rmsd < 10**-4)

    def test_detect_backbone(self):
        geom = TestComponent.RQ_tBu.copy()
        geom.detect_backbone()

        backbone = geom.find('1,2,7-20')
        Me = geom.find('3,21-23')
        tBu = geom.find('4-6,24-59')

        try:
            test_Me = set(geom.find('Me'))
            test_tBu = set(geom.find('tBu'))
            test_backbone = set(geom.find('backbone'))
        except LookupError:
            self.assertTrue(False)

        self.assertTrue(self.is_member(Me, test_Me))
        self.assertTrue(self.is_member(tBu, test_tBu))
        self.assertTrue(self.is_member(backbone, test_backbone))

    def test_minimize_torsion(self):
        geom = TestComponent.benz.copy()
        ref = Component("ref_files/minimize_torsion.xyz")

        geom.substitute('tBu', '12')
        geom.substitute('Ph', '10')
        geom.substitute('OH', '7')

        geom.minimize_sub_torsion()
        rmsd = geom.RMSD(ref)
        self.assertTrue(rmsd < 10**-4)


if __name__ == '__main__':
    unittest.main()
