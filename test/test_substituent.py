#! /usr/bin/env python3
"""Testing for Substituent class"""
import unittest
import numpy as np

from AaronTools.geometry import Geometry
from AaronTools.substituent import Substituent
from AaronTools.test import prefix, TestWithTimer


def check_atom_list(ref, comp):
    rv = True
    for i, j in zip(ref, comp):
        rv &= i.__repr__() == j.__repr__()
    return rv


class TestSubstituent(TestWithTimer):
    COCH3 = Geometry(prefix + 'test_files/COCH3.xyz')
    NO2 = Geometry(prefix + 'test_files/NO2.xyz')
    benz_NO2_Cl = Geometry(prefix + "test_files/benzene_1-NO2_4-Cl.xyz")

    def is_COCH3(self, sub):
        ref = TestSubstituent.COCH3
        self.assertEqual(sub.name, 'COCH3')
        self.assertEqual(sub.comment, 'CF:2,180')
        self.assertEqual(sub.conf_num, 2)
        self.assertEqual(sub.conf_angle, np.deg2rad(180))
        rmsd = ref.RMSD(sub, longsort=True)
        self.assertTrue(rmsd < 10**-8)
        return

    def is_NO2(self, sub):
        ref = TestSubstituent.NO2
        self.assertEqual(sub.name, "NO2")
        self.assertEqual(sub.comment, "CF:2,120")
        self.assertEqual(sub.conf_num, 2)
        self.assertEqual(sub.conf_angle, np.deg2rad(120))
        rmsd = ref.RMSD(sub, sort=True, align=True)
        self.assertTrue(rmsd < 0.1/len(ref.atoms))
        return

    def test_init(self):
        sub = Substituent('COCH3')
        self.is_COCH3(sub)

        sub = Substituent('COCH3', targets=['C', 'O', 'H'])
        self.is_COCH3(sub)
        return

    def test_copy(self):
        sub = Substituent('COCH3')
        sub = sub.copy(name="COCH3")
        self.is_COCH3(sub)
        return

    def test_detect_sub(self):
        mol = TestSubstituent.benz_NO2_Cl

        NO2 = mol.get_fragment('N', 'C', as_object=True)

        sub = Substituent(NO2)
        self.is_NO2(sub)

        NO2 = mol.get_fragment('N', 'C')
        sub = Substituent(NO2)
        self.is_NO2(sub)

    def test_align_to_bond(self):
        mol = TestSubstituent.benz_NO2_Cl
        bond = mol.bond('1', '12')

        sub = Substituent('NO2')
        sub.align_to_bond(bond)

        bond /= np.linalg.norm(bond)
        test_bond = sub.find('N')[0].coords - np.array([0., 0., 0.])
        test_bond /= np.linalg.norm(test_bond)
        self.assertTrue(np.linalg.norm(bond - test_bond) < 10**-8)


if __name__ == "__main__":
    unittest.main()
