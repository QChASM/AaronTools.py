#! /usr/bin/env python3
import unittest
from copy import deepcopy

from AaronTools.catalyst import Catalyst
from AaronTools.component import Component
from AaronTools.geometry import Geometry
from AaronTools.test import prefix, TestWithTimer


class TestCatalyst(TestWithTimer):
    # C:34 L:35-93 K:1,2 F:1-34;1-2;2-34;2-3;3-34
    tm_simple = Catalyst(prefix + "test_files/catalysts/tm_single-lig.xyz")
    tm_multi = Catalyst(prefix + "test_files/catalysts/tm_multi-lig.xyz")
    org_1 = Catalyst(prefix + "test_files/catalysts/org_1.xyz")
    org_tri = Catalyst(prefix + "test_files/catalysts/org_tri.xyz")

    monodentate = Component(prefix + "test_files/ligands/ACN.xyz")
    bidentate = Component(prefix + "test_files/ligands/S-tBu-BOX.xyz")
    tridentate = Component(prefix + "test_files/ligands/squaramide.xyz")

    def validate(self, test, ref):
        t_el = sorted([t.element for t in test.atoms])
        r_el = sorted([r.element for r in ref.atoms])
        if len(t_el) != len(r_el):
            return False

        for t, r in zip(t_el, r_el):
            if t != r:
                return False

        rmsd = ref.RMSD(test, sort=True)
        return rmsd < 10**-5

    def test_init(self):
        self.assertRaises(IOError, Catalyst, prefix +
                          "test_files/R-Quinox-tBu3.xyz")

    def test_detect_components(self):
        def tester(ref, test):
            for comp, r in zip(test.components['ligand'], ref['ligand']):
                good = True
                for i, j in zip(sorted([float(a) for a in comp.atoms]), r):
                    if i != j:
                        good = False
                self.assertTrue(good)
            for comp, r in zip(test.components['substrate'], ref['substrate']):
                good = True
                for i, j in zip(sorted([float(a) for a in comp.atoms]), r):
                    if i != j:
                        good = False
                self.assertTrue(good)
            return

        def printer(test):
            print(test.comment)
            for name, component in test.components.items():
                print(name)
                for comp in component:
                    print("\t", sorted([float(a) for a in comp.atoms]))
            return

        args = []
        ref = {}

        ref['ligand'] = [[float(i) for i in range(35, 94)]]
        ref['substrate'] = [[1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 10.0, 11.0, 12.0,
                             15.0, 16.0, 17.0]]
        ref['substrate'] += [[3.0, 8.0, 9.0, 13.0, 14.0, 18.0, 19.0, 20.0,
                              21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0,
                              29.0, 30.0, 31.0, 32.0, 33.0]]
        args += [(deepcopy(ref), TestCatalyst.tm_simple)]

        ref['ligand'] = [[9.], [10.]]
        ref['ligand'] += [[float(i) for i in range(11, 23)]]
        ref['ligand'] += [[float(i) for i in range(23, 93)]]
        ref['substrate'] = [[float(i) for i in range(1, 8)]]
        args += [(deepcopy(ref), TestCatalyst.tm_multi)]

        ref['ligand'] = [[float(i) for i in range(44, 144)]]
        ref['substrate'] = [[float(i) for i in range(1, 44)]]
        args += [(deepcopy(ref), TestCatalyst.org_1)]

        for a in args:
            tester(*a)

    def test_map_ligand(self):
        monodentate = TestCatalyst.monodentate
        bidentate = TestCatalyst.bidentate
        tridentate = TestCatalyst.tridentate

        tm_simple = TestCatalyst.tm_simple.copy()
        tm_simple.map_ligand(monodentate.copy(), ['35'])
        self.assertTrue(self.validate(
            tm_simple, Geometry(prefix + "ref_files/lig_map_1.xyz")))

        tm_simple = TestCatalyst.tm_simple.copy()
        tm_simple.map_ligand(
            [monodentate.copy(), monodentate.copy()], ['35', '36'])
        self.assertTrue(self.validate(
            tm_simple, Geometry(prefix + "ref_files/lig_map_2.xyz")))

        tm_simple = TestCatalyst.tm_simple.copy()
        tm_simple.map_ligand(bidentate.copy(), ['35', '36'])
        self.assertTrue(self.validate(
            tm_simple, Geometry(prefix + "ref_files/lig_map_3.xyz")))

        org_tri = TestCatalyst.org_tri.copy()
        org_tri.map_ligand(tridentate.copy(), ['30', '28', '58'])
        self.assertTrue(self.validate(
            org_tri, Geometry(prefix + "ref_files/lig_map_4.xyz")))


if __name__ == "__main__":
    unittest.main()
