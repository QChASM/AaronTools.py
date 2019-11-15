#! /usr/bin/env python3
import os
import unittest
from copy import deepcopy

from AaronTools.catalyst import Catalyst
from AaronTools.component import Component
from AaronTools.geometry import Geometry
from AaronTools.test import TestWithTimer, prefix, rmsd_tol, validate
from AaronTools.utils import utils


class TestCatalyst(TestWithTimer):
    # C:34 L:35-93 K:1,2 F:1-34;1-2;2-34;2-3;3-34
    tm_simple = Catalyst(
        os.path.join(prefix, "test_files/catalysts/tm_single-lig.xyz")
    )
    tm_multi = Catalyst(
        os.path.join(prefix, "test_files/catalysts/tm_multi-lig.xyz")
    )
    org_1 = Catalyst(os.path.join(prefix, "test_files/catalysts/org_1.xyz"))
    org_tri = Catalyst(
        os.path.join(prefix, "test_files/catalysts/org_tri.xyz")
    )
    catalysts = [tm_simple, tm_multi, org_1, org_tri]

    monodentate = Component(os.path.join(prefix, "test_files/ligands/ACN.xyz"))
    bidentate = Component(
        os.path.join(prefix, "test_files/ligands/S-tBu-BOX.xyz")
    )
    tridentate = Component(
        os.path.join(prefix, "test_files/ligands/squaramide.xyz")
    )

    def test_init(self):
        self.assertRaises(
            IOError,
            Catalyst,
            os.path.join(prefix, "test_files/R-Quinox-tBu3.xyz"),
        )

    def test_detect_components(self):
        def tester(ref, test):
            for comp, r in zip(test.components["ligand"], ref["ligand"]):
                good = True
                for i, j in zip(sorted([float(a) for a in comp.atoms]), r):
                    if i != j:
                        good = False
                self.assertTrue(good)
            for comp, r in zip(test.components["substrate"], ref["substrate"]):
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

        ref["ligand"] = [[float(i) for i in range(35, 94)]]
        ref["substrate"] = [
            [1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 10.0, 11.0, 12.0, 15.0, 16.0, 17.0]
        ]
        ref["substrate"] += [
            [
                3.0,
                8.0,
                9.0,
                13.0,
                14.0,
                18.0,
                19.0,
                20.0,
                21.0,
                22.0,
                23.0,
                24.0,
                25.0,
                26.0,
                27.0,
                28.0,
                29.0,
                30.0,
                31.0,
                32.0,
                33.0,
            ]
        ]
        args += [(deepcopy(ref), TestCatalyst.tm_simple)]

        ref["ligand"] = [[9.0], [10.0]]
        ref["ligand"] += [[float(i) for i in range(11, 23)]]
        ref["ligand"] += [[float(i) for i in range(23, 93)]]
        ref["substrate"] = [[float(i) for i in range(1, 8)]]
        args += [(deepcopy(ref), TestCatalyst.tm_multi)]

        ref["ligand"] = [[float(i) for i in range(44, 144)]]
        ref["substrate"] = [[float(i) for i in range(1, 44)]]
        args += [(deepcopy(ref), TestCatalyst.org_1)]

        for a in args:
            tester(*a)

    def test_map_ligand(self):
        monodentate = TestCatalyst.monodentate
        tridentate = TestCatalyst.tridentate

        tm_simple = TestCatalyst.tm_simple.copy()
        tm_simple.map_ligand([monodentate, "ACN"], ["35", "36"])
        self.assertTrue(
            validate(
                tm_simple,
                Geometry(os.path.join(prefix, "ref_files/lig_map_2.xyz")),
            )
        )

        tm_simple = TestCatalyst.tm_simple.copy()
        tm_simple.map_ligand("S-tBu-BOX", ["35", "36"])
        self.assertTrue(
            validate(
                tm_simple,
                Geometry(os.path.join(prefix, "ref_files/lig_map_3.xyz")),
            )
        )

        org_tri = TestCatalyst.org_tri.copy()
        org_tri.map_ligand(tridentate, ["30", "28", "58"])
        self.assertTrue(
            validate(
                org_tri,
                Geometry(os.path.join(prefix, "ref_files/lig_map_4.xyz")),
            )
        )

        tm_simple = TestCatalyst.tm_simple.copy()
        tm_simple.map_ligand(monodentate, ["35"])
        self.assertTrue(
            validate(
                tm_simple,
                Geometry(os.path.join(prefix, "ref_files/lig_map_1.xyz")),
            )
        )

    def test_fix_comment(self):
        cat = TestCatalyst.tm_simple.copy()
        cat.write("tmp")
        self.assertEqual(
            cat.comment, "C:34 K:1,5 L:35-93 F:1-2;1-34;2-13;2-34;13-34"
        )
        cat.substitute("Me", "4")
        self.assertEqual(
            cat.comment, "C:37 K:1,5 L:38-96 F:1-2;1-37;2-16;2-37;16-37"
        )

    def test_next_conformer(self):
        total = 24 + 32 + 4
        big_count = 0
        count = 0
        for cat in TestCatalyst.catalysts:
            conf_spec = {}
            for sub in cat.get_substituents():
                conf_spec[sub.atoms[0].name] = 1
            if cat == TestCatalyst.org_1:
                continue
            if count:
                big_count += count
            count = 1
            cat.remove_clash()
            while True:
                if not cat.next_conformer(conf_spec):
                    break
                count += 1
                utils.progress_bar(big_count + count, total)
            subs = sorted(
                [cat.find_substituent(c).name for c in conf_spec.keys()]
            )
            if cat == TestCatalyst.tm_simple:
                self.assertListEqual(subs, sorted(["tBu", "tBu", "tBu", "Et"]))
                self.assertEqual(count, 24)
            elif cat == TestCatalyst.tm_multi:
                self.assertListEqual(
                    subs, sorted(["Ph", "Ph", "Ph", "Ph", "CHO"])
                )
                self.assertEqual(count, 32)
            elif cat == TestCatalyst.org_tri:
                self.assertListEqual(subs, sorted(["Ph", "OMe"]))
                self.assertEqual(count, 4)

        utils.clean_progress_bar()


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestCatalyst("test_init"))
    suite.addTest(TestCatalyst("test_detect_components"))
    suite.addTest(TestCatalyst("test_map_ligand"))
    suite.addTest(TestCatalyst("test_fix_comment"))
    # suite.addTest(TestCatalyst("test_next_conformer"))
    return suite


if __name__ == "__main__":
    # for running specific tests, change test name in suite()
    runner = unittest.TextTestRunner()
    runner.run(suite())
