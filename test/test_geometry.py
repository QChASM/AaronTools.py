#!/usr/bin/env python3
import json
import os
import unittest
from copy import copy

import AaronTools
import numpy as np
from AaronTools.atoms import Atom
from AaronTools.component import Component
from AaronTools.fileIO import FileReader
from AaronTools.geometry import Geometry
from AaronTools.ring import Ring
from AaronTools.substituent import Substituent
from AaronTools.test import TestWithTimer, prefix, rmsd_tol, validate


def is_close(a, b, tol=10 ** -8, debug=False):
    n = None
    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        n = np.linalg.norm(a - b)
    elif isinstance(a, np.ndarray):
        n = np.linalg.norm(a) - b
    elif isinstance(b, np.ndarray):
        n = a - np.linalg.norm(b)
    else:
        n = a - b
    if debug:
        print(abs(n))
    return abs(n) < tol


class TestGeometry(TestWithTimer):
    benz_NO2_Cl = os.path.join(prefix, "test_files", "benzene_1-NO2_4-Cl.xyz")
    benzene_oniom = os.path.join(prefix, "test_files", "benzene_oniom.xyz")
    benz_NO2_Cl_conn = [
        "2,6,12",
        "1,3,7",
        "2,4,8",
        "3,5,11",
        "4,6,9",
        "1,5,10",
        "2",
        "3",
        "5",
        "6",
        "4",
        "1,13,14",
        "12",
        "12",
    ]
    benz_NO2_Cl_conn = [i.split(",") for i in benz_NO2_Cl_conn]
    benzene = os.path.join(prefix, "test_files", "benzene.xyz")
    pyridine = os.path.join(prefix, "test_files", "pyridine.xyz")
    pentane = os.path.join(prefix, "test_files", "pentane.xyz")
    naphthalene = os.path.join(prefix, "ref_files", "naphthalene.xyz")
    tetrahydronaphthalene = os.path.join(
        prefix, "ref_files", "tetrahydronaphthalene.xyz"
    )
    pyrene = os.path.join(prefix, "ref_files", "pyrene.xyz")
    benz_Cl = os.path.join(prefix, "test_files", "benzene_4-Cl.xyz")
    benz_OH_Cl = os.path.join(prefix, "test_files", "benzene_1-OH_4-Cl.xyz")
    benz_Ph_Cl = os.path.join(prefix, "test_files", "benzene_1-Ph_4-Cl.xyz")
    Et_NO2 = os.path.join(prefix, "test_files", "Et_1-NO2.xyz")
    cat = os.path.join(prefix, "test_files", "catalysts", "tm_single-lig.xyz")

    tm_simple = os.path.join(
        prefix, "test_files", "catalysts/tm_single-lig.xyz"
    )
    tm_multi = os.path.join(
        prefix, "test_files", "catalysts", "tm_multi-lig.xyz"
    )
    org_1 = os.path.join(prefix, "test_files", "catalysts", "org_1.xyz")
    org_tri = os.path.join(prefix, "test_files", "catalysts", "org_tri.xyz")
    ptco4 = os.path.join(prefix, "test_files", "ptco4.xyz")
    catalysts = [tm_simple, tm_multi, org_1, org_tri, ptco4]

    monodentate = os.path.join(prefix, "test_files", "ligands", "ACN.xyz")
    bidentate = os.path.join(prefix, "test_files", "ligands", "S-tBu-BOX.xyz")
    tridentate = os.path.join(
        prefix, "test_files", "ligands", "squaramide.xyz"
    )

    lig_1 = os.path.join(prefix, "test_files", "lig_1.xyz")
    lig_2 = os.path.join(prefix, "test_files", "lig_2.xyz")
    pd_1 = os.path.join(prefix, "test_files", "pd_complex_1.com")
    pd_2 = os.path.join(prefix, "test_files", "pd_complex_2.com")

    def test_init(self):
        ref = FileReader(TestGeometry.benz_NO2_Cl)
        conn_valid = TestGeometry.benz_NO2_Cl_conn
        rank_valid = [8, 7, 5, 4, 5, 7, 3, 1, 1, 3, 0, 6, 2, 2]

        for i, a in enumerate(ref.atoms):
            a.connected = set(conn_valid[i])
            a._rank = rank_valid[i]

        mol = Geometry(TestGeometry.benz_NO2_Cl)

        # name
        self.assertEqual(mol.name, ref.name)
        # comment
        self.assertEqual(mol.comment, ref.comment)
        # atoms
        for a, b in zip(ref.atoms, mol.atoms):
            for attr in a.__dict__:
                if attr == "file_type":
                    continue
                if attr == "connected":
                    tmpa = [int(c) for c in a.connected]
                    tmpb = [int(c.name) for c in b.connected]
                    self.assertSequenceEqual(sorted(tmpa), sorted(tmpb))
                    continue
                if attr == "_rank":
                    continue
                if attr == "constraint":
                    # this is checked in test_parse_comment
                    continue
                try:
                    self.assertEqual(a.__dict__[attr], b.__dict__[attr])
                except ValueError:
                    self.assertSequenceEqual(
                        sorted(a.__dict__[attr]), sorted(b.__dict__[attr])
                    )

        # test blank
        blank = Geometry()
        self.assertEqual(blank.name, "")
        self.assertEqual(blank.comment, "")
        self.assertEqual(blank.atoms, [])

    def test_attribute_access(self):
        mol = Geometry(TestGeometry.benzene)
        # stack coords
        coords = mol._stack_coords()
        self.assertEqual(coords.shape, (12, 3))
        # elements
        elements = mol.elements
        self.assertEqual(len(elements), 12)
        self.assertEqual(elements[0], "C")
        # coords
        coords = mol.coords
        self.assertEqual(coords.shape, (12, 3))

    # utilities
    def test_equality(self):
        mol = Geometry(TestGeometry.benz_NO2_Cl)
        benz = Geometry(TestGeometry.benzene)
        # same object should be equal
        self.assertEqual(mol, mol)
        # copy should be equal
        self.assertEqual(mol, mol.copy())
        # different molecules should be unequal
        self.assertNotEqual(mol, benz)

    def test_add_sub_iter_len(self):
        ref = Geometry(TestGeometry.benz_OH_Cl)
        mol = Geometry(TestGeometry.benzene)
        mol.debug = True
        OH = ref.find(["12", "13"])
        Cl = ref.find(["11"])
        mol -= mol.find(["11", "12"])
        mol += Cl
        mol += OH
        mol.refresh_connected()
        mol.refresh_ranks()
        self.assertTrue(validate(mol, ref, thresh="loose"))

    def test_find_atom(self):
        geom = Geometry(TestGeometry.benz_NO2_Cl)
        geom.atoms[0].add_tag("find_me")

        # find a specific atom
        a = geom.find(geom.atoms[0])
        self.assertEqual(a, geom.atoms[0:1])

        # find by tag
        a = geom.find("find_me")
        self.assertEqual(a, geom.atoms[0:1])

        # find by name
        a = geom.find("1")
        self.assertEqual(a, geom.atoms[0:1])

        # find by list-style name
        a = geom.find("1,2-5,12")
        self.assertSequenceEqual(
            a, geom.atoms[0:1] + geom.atoms[1:5] + geom.atoms[11:12]
        )

        # find using tag and name
        a = geom.find("find_me", "1")
        self.assertEqual(a, geom.atoms[0:1])

        b = geom.find(["find_me", "2"])
        self.assertEqual(b, geom.atoms[0:2])

        c = geom.find(["1", "2"], "find_me")
        self.assertEqual(c, geom.atoms[0:1])

        d = geom.find(["2", "3"], "find_me")
        self.assertEqual(d, [])

        # find all Carbons
        a = geom.find("C")
        b = []
        for i in geom.atoms:
            if i.element == "C":
                b += [i]
        self.assertSequenceEqual(a, b)

        # raise error when atom not found
        self.assertRaises(LookupError, geom.find, "definitely not in here")

    def test_refresh_connected(self):
        # refresh_connected should be run upon creation
        mol = Geometry(TestGeometry.benz_NO2_Cl)
        conn_valid = TestGeometry.benz_NO2_Cl_conn
        for a, b in zip(mol.atoms, conn_valid):
            tmpa = [int(c.name) for c in a.connected]
            tmpb = [int(c) for c in b]
            self.assertSequenceEqual(sorted(tmpa), sorted(tmpb))

        # old connectivity shouldn't remain in the set
        mol.atoms[0].connected.add(Atom())
        old = mol.atoms[0].connected
        mol.refresh_connected()
        self.assertTrue(len(old) - len(mol.atoms[0].connected) == 1)

    def test_canonical_rank(self):
        pentane = Geometry(
            os.path.join(prefix, "test_files", "pentane.xyz"),
            refresh_ranks=False
        )
        pentane_rank = [1, 3, 4, 2, 0]
        test_rank = pentane.canonical_rank(heavy_only=True)
        self.assertSequenceEqual(test_rank, pentane_rank)

        mol = Geometry(os.path.join(prefix, "test_files", "6a2e5am1hex.xyz"))
        mol_rank = [8, 6, 5, 7, 2, 1, 4, 3, 11, 0, 10, 9]
        test_rank = mol.canonical_rank(heavy_only=True, invariant=False)
        self.assertSequenceEqual(test_rank, mol_rank)

    def test_flag(self):
        geom = Geometry(TestGeometry.benz_NO2_Cl)

        # freeze all
        test = geom.copy()
        test.freeze()
        for a in test.atoms:
            self.assertTrue(a.flag)

        # freeze some
        test = geom.copy()
        test.freeze("C")
        for a in test.atoms:
            if a.element == "C":
                self.assertTrue(a.flag)
            else:
                self.assertFalse(a.flag)

        geom.freeze()
        # relax all
        test = geom.copy()
        test.relax()
        for a in test.atoms:
            self.assertFalse(a.flag)

        # relax some
        test = geom.copy()
        test.relax("C")
        for a in test.atoms:
            if a.element == "C":
                self.assertFalse(a.flag)
            else:
                self.assertTrue(a.flag)

    def test_update_geometry(self):
        test = Geometry(TestGeometry.benz_NO2_Cl)

        # using coordinate matrix to update
        ref = test.copy()
        ref.coord_shift([-10, 0, 0])
        tmp = test._stack_coords()
        for t in tmp:
            tmp = tmp - np.array([-10, 0, 0])
        test.update_geometry(tmp)
        self.assertTrue(validate(test, ref))

        # using file
        ref.coord_shift([10, 0, 0])
        test.update_geometry(TestGeometry.benz_NO2_Cl)
        self.assertTrue(validate(test, ref))

    def test_near(self):
        def compare(test, ref):
            self.assertTrue(len(test) == len(set(test)))
            test = set(test)
            for a in ref:
                self.assertTrue(a in test)
                test.discard(a)
            self.assertTrue(len(test) == 0)

        geom = Geometry(self.benz_NO2_Cl)

        # atoms within 3A of origin
        test = geom.get_near([0, 0, 0], 3)
        ref = geom.find(["2", "3", "4", "5", "6", "8", "9", "11"])
        compare(test, ref)
        # atoms within 1A of x-axis
        test = geom.get_near(["*", 0, 0], 1)
        ref = geom.find(["3", "4"])
        compare(test, ref)
        # atoms within 0.5A of xy-plane
        test = geom.get_near(["*", "*", 0], 0.5)
        ref = geom.find(["1", "4", "11", "12", "13", "14"])
        compare(test, ref)

        # atoms within 2A of atom 1
        test = geom.get_near(geom.atoms[0], 2)
        ref = geom.find(["2", "6", "12"])
        compare(test, ref)
        # ...including atom 1
        test = geom.get_near(geom.atoms[0], 2, include_ref=True)
        ref = geom.find(["1", "2", "6", "12"])
        compare(test, ref)
        # atoms within 2 bonds of atom 1
        test = geom.get_near(geom.atoms[0], 2, by_bond=True)
        ref = geom.find(["2", "6", "12", "3", "7", "5", "10", "13", "14"])
        compare(test, ref)
        # atoms within 1 bond of atoms 1 and 2
        test = geom.get_near(geom.atoms[:2], 1, by_bond=True)
        ref = geom.find(["6", "12", "3", "7"])
        compare(test, ref)
        # ...including starting atoms
        test = geom.get_near(geom.atoms[:2], 1, by_bond=True, include_ref=True)
        ref = geom.find(["6", "12", "3", "7", "2", "1"])
        compare(test, ref)

    def test_compare_connectivity(self):
        geom = Geometry(TestGeometry.cat)
        ref = Geometry(TestGeometry.cat)

        # no formed/broken
        broken, formed = geom.compare_connectivity(ref)
        self.assertTrue(len(broken) == 0)
        self.assertTrue(len(formed) == 0)
        # broken
        geom.change_distance("10", "15", dist=1, adjust=True)
        geom.refresh_connected()
        broken, formed = geom.compare_connectivity(ref)
        self.assertTrue(len(broken) == 1)
        self.assertTrue(len(formed) == 0)
        self.assertSetEqual(broken, set([("10", "15")]))
        # formed
        ref.change_distance("10", "15", dist=1, adjust=True)
        ref.refresh_connected()
        geom.change_distance("10", "15", dist=-1, adjust=True)
        geom.refresh_connected()
        broken, formed = geom.compare_connectivity(ref)
        self.assertTrue(len(broken) == 0)
        self.assertTrue(len(formed) == 1)
        self.assertSetEqual(formed, set([("10", "15")]))
        # broken and formed
        geom.change_distance("20", "29", dist=1, adjust=True)
        geom.refresh_connected()
        broken, formed = geom.compare_connectivity(ref)
        self.assertTrue(len(broken) == 1)
        self.assertTrue(len(formed) == 1)
        self.assertSetEqual(broken, set([("20", "29")]))
        self.assertSetEqual(formed, set([("10", "15")]))

    def test_examine_constraints(self):
        cat = Geometry(TestGeometry.cat)
        rv = cat.examine_constraints()
        self.assertSequenceEqual(rv, [])

        cat.change_distance(cat.atoms[0], cat.atoms[1], dist=0.5, adjust=True)
        rv = cat.examine_constraints()
        self.assertSequenceEqual(rv, [(0, 1, -1)])

        cat.change_distance(cat.atoms[0], cat.atoms[1], dist=-1.0, adjust=True)
        rv = cat.examine_constraints()
        self.assertSequenceEqual(rv, [(0, 1, 1)])

        cat.change_distance(cat.atoms[1], cat.atoms[2], dist=0.5, adjust=True)
        rv = cat.examine_constraints()
        self.assertSequenceEqual(rv, [(0, 1, 1), (1, 2, -1)])

    def test_detect_components(self):
        test = {}
        for cat in TestGeometry.catalysts:
            cat = Geometry(cat)
            cat.detect_components()
            for comp in cat.components:
                test[os.path.basename(comp.name)] = sorted(
                    [int(float(c)) for c in comp]
                )

        with open(
            os.path.join(prefix, "ref_files", "detect_components.json")
        ) as f:
            ref = json.load(f)
        self.assertDictEqual(test, ref)

    def test_fix_comment(self):
        cat = Geometry(TestGeometry.tm_simple)
        cat.fix_comment()
        self.assertEqual(
            cat.comment,
            "C:1 K:2,3;14;39,40 F:1-2;1-3;1-14;2-3;2-14 L:2-13;14-34;35-93",
        )
        cat.substitute("Me", "4")
        self.assertEqual(
            cat.comment,
            "C:1 K:2,3;14;39,40 F:1-2;1-3;1-14;2-3;2-14 L:2-13;14-34;35-93",
        )

    # geometry measurement
    def test_angle(self):
        mol = Geometry(TestGeometry.benz_NO2_Cl)
        angle = mol.angle("13", "12", "14")
        self.assertTrue(is_close(np.rad2deg(angle), 124.752, 10 ** -2))

    def test_dihedral(self):
        mol = Geometry(TestGeometry.benz_NO2_Cl)
        dihedral = mol.dihedral("13", "12", "1", "6")
        self.assertTrue(is_close(np.rad2deg(dihedral), 45.023740, 10 ** -5))

    def test_COM(self):
        mol = Geometry(TestGeometry.benz_NO2_Cl)

        # all atoms
        com = mol.COM(mass_weight=True)
        self.assertTrue(
            is_close(com, [-1.41347592e00, -1.35049427e00, 9.24797359e-04])
        )

        # only carbons
        com = mol.COM(targets="C")
        self.assertTrue(
            is_close(com, [-1.27963500e00, -1.11897500e00, 8.53333333e-04])
        )
        # mass weighting COM shouldn't change anythin if they are all C
        com = mol.COM(targets="C", mass_weight=True)
        self.assertTrue(
            is_close(com, [-1.27963500e00, -1.11897500e00, 8.53333333e-04])
        )

        # only heavy atoms
        com = mol.COM(heavy_only=True, mass_weight=True)
        self.assertTrue(
            is_close(com, [-1.41699980e00, -1.35659562e00, 9.31512531e-04])
        )

    def test_RMSD(self):
        ref = Geometry(TestGeometry.benz_NO2_Cl)

        # RMSD of copied object should be 0
        test = ref.copy()
        self.assertTrue(validate(test, ref))

        # RMSD of shifted copy should be 0
        test = ref.copy()
        test.coord_shift([1, 2, 3])
        self.assertTrue(validate(test, ref))

        # RMSD of rotated copy should be 0
        test = ref.copy()
        test.rotate([1, 2, 3], 2.8)
        self.assertTrue(validate(test, ref))

        # RMSD of two different structures should not be 0
        test = Geometry(TestGeometry.pentane)
        self.assertFalse(validate(test, ref))

        # RMSD of similar molecule
        test = Geometry(TestGeometry.benzene)
        res = ref.RMSD(test, targets="C", ref_targets="C")
        self.assertTrue(res < rmsd_tol(ref))

    def test_vbur_MC(self):
        """
        tests % volume buried (MC integration)
        uses Monte Carlo integration, so it this fails, run it again
        still figuring out how reliable it is
        """

        # import cProfile
        # 
        # profile = cProfile.Profile()
        # profile.enable()

        geom = Geometry(os.path.join(prefix, "ref_files", "lig_map_3.xyz"))
        vbur = geom.percent_buried_volume(method="MC")
        if not np.isclose(vbur, 86.0, atol=0.35):
            print("V_bur =", vbur, "expected:", 86.0)
        self.assertTrue(np.isclose(vbur, 86.0, atol=0.35))

        # a few synthetic tests
        geom2 = Geometry(os.path.join(prefix, "ref_files", "vbur.xyz"))
        vbur = geom2.percent_buried_volume(
            method="MC", scale=1 / 1.1, radius=3
        )
        if not np.isclose(vbur, 100.0 / 27, atol=0.2):
            print("V_bur =", vbur, "expected:", 100.0 / 27)
        self.assertTrue(np.isclose(vbur, 100.0 / 27, atol=0.2))

        geom3 = Geometry(os.path.join(prefix, "ref_files", "vbur2.xyz"))
        vbur = geom2.percent_buried_volume(
            method="MC", scale=1 / 1.1, radius=4
        )
        if not np.isclose(vbur, 100.0 / 64, atol=0.2):
            print("V_bur =", vbur, "expected:", 100.0 / 64)
        self.assertTrue(np.isclose(vbur, 100.0 / 64, atol=0.2))

        # profile.disable()
        # profile.print_stats()

    def test_vbur_lebedev(self):
        """
        tests % volume buried (Lebedev integration)
        """
        # 20, 1454
        geom = Geometry(os.path.join(prefix, "ref_files", "lig_map_3.xyz"))
        vbur = geom.percent_buried_volume(
            method="lebedev", rpoints=20, apoints=1454
        )
        if not np.isclose(vbur, 86.0, atol=0.05):
            print("V_bur =", vbur, "expected:", 86.0)
        self.assertTrue(np.isclose(vbur, 86.0, atol=0.05))

        # 32, 974
        vbur = geom.percent_buried_volume(
            method="lebedev", rpoints=32, apoints=974
        )
        if not np.isclose(vbur, 86.0, atol=0.15):
            print("V_bur =", vbur, "expected:", 86.0)
        self.assertTrue(np.isclose(vbur, 86.0, atol=0.15))

        # 64, 590
        vbur = geom.percent_buried_volume(
            method="lebedev", rpoints=64, apoints=590
        )
        if not np.isclose(vbur, 86.0, atol=0.05):
            print("V_bur =", vbur, "expected:", 86.0)
        self.assertTrue(np.isclose(vbur, 86.0, atol=0.05))

        # 75, 302
        vbur = geom.percent_buried_volume(
            method="lebedev", rpoints=75, apoints=302
        )
        if not np.isclose(vbur, 86.0, atol=0.1):
            print("V_bur =", vbur, "expected:", 86.0)
        self.assertTrue(np.isclose(vbur, 86.0, atol=0.1))

        # 99, 194
        vbur = geom.percent_buried_volume(
            method="lebedev", rpoints=99, apoints=194
        )
        if not np.isclose(vbur, 86.0, atol=0.5):
            print("V_bur =", vbur, "expected:", 86.0)
        self.assertTrue(np.isclose(vbur, 86.0, atol=0.5))

        # 127, 110
        vbur = geom.percent_buried_volume(
            method="lebedev", rpoints=127, apoints=110
        )
        if not np.isclose(vbur, 86.0, atol=0.75):
            print("V_bur =", vbur, "expected:", 86.0)
        self.assertTrue(np.isclose(vbur, 86.0, atol=0.75))

        # 20, 2030
        vbur = geom.percent_buried_volume(
            method="lebedev", rpoints=20, apoints=2030
        )
        if not np.isclose(vbur, 86.0, atol=0.1):
            print("V_bur =", vbur, "expected:", 86.0)
        self.assertTrue(np.isclose(vbur, 86.0, atol=0.1))

        # 20, 2702
        vbur = geom.percent_buried_volume(
            method="lebedev", rpoints=20, apoints=2702
        )
        if not np.isclose(vbur, 86.0, atol=0.15):
            print("V_bur =", vbur, "expected:", 86.0)
        self.assertTrue(np.isclose(vbur, 86.0, atol=0.15))

        # 20, 5810
        vbur = geom.percent_buried_volume(
            method="lebedev", rpoints=20, apoints=5810
        )
        if not np.isclose(vbur, 86.0, atol=0.1):
            print("V_bur =", vbur, "expected:", 86.0)
        self.assertTrue(np.isclose(vbur, 86.0, atol=0.1))

        geom2 = Geometry(os.path.join(prefix, "ref_files", "vbur.xyz"))
        vbur = geom2.percent_buried_volume(
            method="lebedev", rpoints=32, apoints=974, scale=1 / 1.1, radius=2
        )
        if not np.isclose(vbur, 100.0 / 8, atol=0.1):
            print("V_bur =", vbur, "expected:", 100.0 / 8)
        self.assertTrue(np.isclose(vbur, 100.0 / 8, atol=0.1))

    # geometry manipulation
    def test_get_fragment(self):
        mol = Geometry(TestGeometry.benz_NO2_Cl)

        # get Cl using name
        frag = mol.get_fragment("11", "4", copy=False)
        v_frag = mol.atoms[10:11]
        self.assertSequenceEqual(frag, v_frag)

        # get ring without NO2 using atoms
        frag = mol.get_fragment(mol.atoms[0], mol.atoms[11], copy=False)
        v_frag = mol.atoms[:11]
        self.assertSequenceEqual(sorted(frag), sorted(v_frag))

        # get fragment as Geometry()
        frag = mol.get_fragment(mol.atoms[0], mol.atoms[11], as_object=True)
        self.assertIsInstance(frag, Geometry)

    def test_remove_fragment(self):
        benzene = Geometry(TestGeometry.benzene)
        benz_Cl = Geometry(TestGeometry.benz_Cl)
        Et_NO2 = Geometry(TestGeometry.Et_NO2)

        # remove NO2 group using atom name
        mol = Geometry(TestGeometry.benz_NO2_Cl)
        mol.remove_fragment("12", "1")
        self.assertEqual(mol, benz_Cl)
        # remove Cl using elements
        mol.remove_fragment("Cl", "C")
        self.assertEqual(mol, benzene)

        # multiple start atoms
        mol = Geometry(TestGeometry.benz_NO2_Cl)
        mol.remove_fragment(["3", "6"], ["2", "1"])
        self.assertEqual(mol, Et_NO2)

    def test_coord_shift(self):
        benzene = Geometry(TestGeometry.benzene)
        mol = Geometry(TestGeometry.benzene)

        # shift all atoms
        vector = np.array([0, 3.2, -1.0])
        for a in benzene.atoms:
            a.coords += vector
        mol.coord_shift([0, 3.2, -1.0])
        self.assertTrue(np.linalg.norm(benzene.coords - mol.coords) == 0)

        # shift some atoms
        vector = np.array([0, -3.2, 1.0])
        for a in benzene.atoms[0:5]:
            a.coords += vector
        mol.coord_shift([0, -3.2, 1.0], [str(i) for i in range(1, 6)])
        self.assertTrue(np.linalg.norm(benzene.coords - mol.coords) == 0)

    def test_change_distance(self):
        def validate_distance(before, after, moved):
            dist = np.linalg.norm(after.coords - before.coords)
            return abs(dist - moved)

        threshold = 10 ** (-8)
        mol = Geometry(TestGeometry.benz_NO2_Cl)
        a1 = mol.atoms[0]  # C1
        a3 = mol.atoms[5]  # attached to C1
        a2 = mol.atoms[11]  # N12
        a4 = mol.atoms[12]  # attached to N12

        # set distance and move fragments
        a3_before = copy(a3)
        a4_before = copy(a4)
        dist_before = np.linalg.norm(a1.coords - a2.coords)
        mol.change_distance(a1, a2, dist=2.00)
        dist = np.linalg.norm(a1.coords - a2.coords)
        self.assertTrue(is_close(dist, 2.00, threshold))
        self.assertTrue(validate_distance(a3_before, a3, dist - dist_before))
        self.assertTrue(validate_distance(a4_before, a4, dist - dist_before))

        # adjust_distance and move fragments
        a3_before = copy(a3)
        a4_before = copy(a4)
        dist_before = np.linalg.norm(a1.coords - a2.coords)
        mol.change_distance(a1, a2, dist=-0.5, adjust=True)
        self.assertTrue(is_close(a1.dist(a2), 1.5, threshold))
        self.assertTrue(validate_distance(a3_before, a3, -0.5))
        self.assertTrue(validate_distance(a4_before, a4, -0.5))

        # set and fix atom 1
        a1_before = copy(a1)
        mol.change_distance(a1, a2, dist=2.00, fix=1)
        self.assertTrue(is_close(a1.dist(a2), 2.00, threshold))
        self.assertTrue(a1_before.dist(a1) < threshold)

        # adjust and fix atom 2
        a2_before = copy(a2)
        mol.change_distance(a1, a2, dist=-0.5, adjust=True, fix=2)
        self.assertTrue(is_close(a1.dist(a2), 1.5, threshold))
        self.assertTrue(a2_before.dist(a2) < threshold)

        # set and don't move fragments
        a3_before = copy(a3)
        a4_before = copy(a4)
        dist_before = np.linalg.norm(a1.coords - a2.coords)
        mol.change_distance(a1, a2, dist=2.00, as_group=False)
        dist = np.linalg.norm(a1.coords - a2.coords)
        self.assertTrue(is_close(a1.dist(a2), 2.00, threshold))
        self.assertTrue(validate_distance(a3_before, a3, dist - dist_before))
        self.assertTrue(validate_distance(a4_before, a4, dist - dist_before))

    def test_change_angle(self):
        def diff(a, b):
            return abs(a - b)

        mol = Geometry(TestGeometry.benz_NO2_Cl)

        # set angle
        mol.change_angle("13", "12", "14", np.pi / 2, fix=1)
        angle = mol.angle("13", "12", "14")
        self.assertTrue(diff(np.rad2deg(angle), 90) < 10 ** -8)

        # change angle
        mol.change_angle(
            "13", "12", "14", 30, fix=3, adjust=True, radians=False
        )
        angle = mol.angle("13", "12", "14")
        self.assertTrue(diff(np.rad2deg(angle), 120) < 10 ** -8)

        mol.change_angle("13", "12", "14", -30, adjust=True, radians=False)
        angle = mol.angle("13", "12", "14")
        self.assertTrue(diff(np.rad2deg(angle), 90) < 10 ** -8)

    def test_change_dihedral(self):
        mol = Geometry(TestGeometry.benz_NO2_Cl)
        atom_args = ("13", "12", "1", "6")
        original_dihedral = mol.dihedral(*atom_args)

        # adjust dihedral by 30 degrees
        mol.change_dihedral(*atom_args, 30, radians=False, adjust=True)
        self.assertTrue(
            is_close(
                mol.dihedral(*atom_args), original_dihedral + np.deg2rad(30)
            )
        )

        # set dihedral to 60 deg
        mol.change_dihedral(*atom_args, 60, radians=False)
        self.assertTrue(is_close(mol.dihedral(*atom_args), np.deg2rad(60)))

        # adjust using just two atoms
        mol.change_dihedral("12", "1", -30, radians=False, adjust=True)
        self.assertTrue(is_close(mol.dihedral(*atom_args), np.deg2rad(30)))

    def test_substitute(self):
        ref = Geometry(TestGeometry.benz_NO2_Cl)
        mol = Geometry(TestGeometry.benzene)

        mol.substitute(Substituent("NO2"), "12")
        mol.substitute(Substituent("Cl"), "11")

        self.assertTrue(validate(mol, ref))

    def test_close_ring(self):
        mol = Geometry(TestGeometry.benzene)

        ref1 = Geometry(TestGeometry.naphthalene)
        mol1 = mol.copy()
        mol1.ring_substitute(["7", "8"], Ring("benzene"))
        self.assertTrue(validate(mol1, ref1, thresh="loose"))

        ref2 = Geometry(TestGeometry.tetrahydronaphthalene)
        mol2 = mol.copy()
        mol2.ring_substitute(["7", "8"], Ring("cyclohexane"))
        rmsd = mol2.RMSD(ref2, align=True, sort=True)
        self.assertTrue(rmsd < rmsd_tol(ref2, superLoose=True))

        mol3 = Geometry(TestGeometry.naphthalene)
        ref3 = Geometry(TestGeometry.pyrene)
        targets1 = mol3.find(["9", "15"])
        targets2 = mol3.find(["10", "16"])
        mol3.ring_substitute(targets1, Ring("benzene"))
        mol3.ring_substitute(targets2, Ring("benzene"))
        rmsd = mol3.RMSD(ref3, align=True, sort=True)
        self.assertTrue(rmsd < rmsd_tol(ref3, superLoose=True))

    def test_change_element(self):
        mol = Geometry(TestGeometry.benzene)

        ref = Geometry(TestGeometry.pyridine)
        mol.change_element("1", "N", adjust_hydrogens=True)
        self.assertTrue(validate(mol, ref, thresh="loose"))

    def test_map_ligand(self):
        monodentate = Component(TestGeometry.monodentate)
        tridentate = Component(TestGeometry.tridentate)
        debug = False

        # import cProfile
        #
        # profile = cProfile.Profile()
        # profile.enable()

        """
        #TODO: get a reference file for this
        # two monodentate -> bidentate
        ptco4 = TestGeometry.ptco4.copy()
        ptco4.map_ligand('EDA', ["3", "5"])
        """

        # bidentate -> monodentate, none
        ref = Geometry(os.path.join(prefix, "ref_files", "lig_map_1.xyz"))
        tm_simple = Geometry(TestGeometry.tm_simple)
        tm_simple.map_ligand(monodentate.copy(), ["35"])
        self.assertTrue(
            validate(
                tm_simple, ref, heavy_only=True, thresh="loose", debug=debug
            )
        )

        # bidentate -> two monodentate
        ref = Geometry(os.path.join(prefix, "ref_files", "lig_map_2.xyz"))
        tm_simple = Geometry(TestGeometry.tm_simple)
        tm_simple.map_ligand([monodentate.copy(), "ACN"], ["35", "36"])
        self.assertTrue(
            validate(
                tm_simple, ref, thresh="loose", heavy_only=True, debug=debug
            )
        )

        # bidentate -> bidentate
        ref = Geometry(os.path.join(prefix, "ref_files", "lig_map_3.xyz"))
        tm_simple = Geometry(TestGeometry.tm_simple)
        tm_simple.map_ligand("S-tBu-BOX", ["35", "36"])
        self.assertTrue(
            validate(
                tm_simple, ref, thresh="loose", heavy_only=True, debug=debug
            )
        )

        # tridentate -> tridentate
        ref = Geometry(os.path.join(prefix, "ref_files", "lig_map_4.xyz"))
        org_tri = Geometry(TestGeometry.org_tri)
        org_tri.map_ligand(tridentate, ["30", "28", "58"])
        self.assertTrue(
            validate(
                org_tri, ref, thresh="loose", heavy_only=True, debug=debug
            )
        )

        # tridentate -> monodentate + bidentate -> tridentate
        ref = Geometry(os.path.join(prefix, "ref_files", "lig_map_6.xyz"))
        org_tri = Geometry(TestGeometry.org_tri)
        org_tri.map_ligand(["EDA", "ACN"], ["30", "28", "58"])
        self.assertTrue(
            validate(
                org_tri, ref, thresh="loose", heavy_only=True, debug=debug
            )
        )

        ref = Geometry(os.path.join(prefix, "ref_files", "lig_map_7.xyz"))
        org_tri = Geometry(os.path.join(prefix, "ref_files", "lig_map_6.xyz"))
        org_tri.map_ligand(tridentate, ["10", "11", "2"])
        self.assertTrue(
            validate(
                org_tri, ref, thresh="loose", heavy_only=True, debug=debug
            )
        )

        # bidentate -> two bulky monodentate
        ref = Geometry(os.path.join(prefix, "ref_files", "lig_map_5.xyz"))
        tm_simple = Geometry(TestGeometry.tm_simple)
        tm_simple.map_ligand(["iPr-NC3C"] * 2, ["35", "36"])
        self.assertTrue(
            validate(
                tm_simple, ref, thresh="loose", heavy_only=True, debug=debug
            )
        )

        # profile.disable()
        # profile.print_stats()

    def test_get_gaff_geom(self):
        ref = Geometry(TestGeometry.benzene_oniom)
        mol = Geometry(TestGeometry.benz_NO2_Cl)
        mol2 = mol.get_gaff_geom()
        self.assertEqual(ref,mol2)

    def test_get_aromatic_atoms(self):
        mol = Geometry(TestGeometry.benzene)
        out,out2,out3 = mol.get_aromatic_atoms(mol.atoms, return_rings=False,return_h=False)
        self.assertTrue(all([atom in mol.find('C') for atom in out]))

    def test_define_layer(self):
        g = Geometry(TestGeometry.pd_1)
        h = Geometry(TestGeometry.pd_2)
        g.define_layer("H", g.atoms[0], 4)
        g.define_layer("L", "!H", 5)
        counter = 0
        for index, atom in enumerate(g.atoms):
            if atom.layer == h.atoms[index].layer:
                counter += 1
#            elif atom.layer != h.atoms[index].layer:
#                print(atom, h.atoms[index])
        self.assertEqual(counter, len(g.atoms))
#        self.assertTrue(g == h)

    def test_detect_solvent(self):
        g = Geometry(TestGeometry.pd_1)
        mol_1 = Geometry(structure=[Atom(name = "74", element = "C", coords =[6.040704, 0.007857, 0.264571]),
                Atom(name="75", element="H", coords= [4.952884, 0.082218, 0.304678]),
                Atom(name="77", element="Cl", coords= [ 6.551478, -1.353073,  1.318184]),
                Atom(name="78", element="Cl", coords= [ 6.507952, -0.297483, -1.439601]),
                Atom(name="76", element="Cl", coords= [6.737007, 1.550031, 0.844314])])

        mol_2 = Geometry(structure=[Atom(name="79", element="C" , coords= [ 8.232315, -6.21151 , -0.70706 ]),
                Atom(name="80", element="H" , coords= [ 9.302822, -6.374525, -0.627481]),
                Atom(name="82", element="Cl" , coords= [ 7.901812, -4.527103, -0.209528]),
                Atom(name="83", element="Cl" , coords= [ 7.773422, -6.483229, -2.416086]),
                Atom(name="81", element="Cl" , coords= [ 7.422991, -7.380789 , 0.374106])])

        mol_3 = Geometry(structure=[Atom(name="84", element= "C" , coords= [-9.068698, -1.81313  , 2.409807]),
                Atom(name="85", element= "H" , coords= [-10.069529,  -1.900536  , 2.821704]),
                Atom(name="86", element= "Cl" , coords= [-8.062186, -0.920557 , 3.571119]),
                Atom(name="88", element= "Cl" , coords= [-9.206153, -0.941428 , 0.850172]),
                Atom(name="87", element= "Cl" , coords= [-8.454248, -3.48083  , 2.152553])])

        mol_4 = Geometry(structure=[Atom(name="89", element= "C" , coords= [ 1.813612, 10.176062, -1.144781]),
                Atom(name="90", element= "H" , coords= [ 1.710907 , 9.10081 , -1.029659]),
                Atom(name="92", element= "Cl" , coords= [ 0.195889, 10.847618, -1.510565]),
                Atom(name="91", element= "Cl" , coords= [ 2.953246, 10.474472, -2.493735]),
                Atom(name="93", element= "Cl" , coords= [ 2.451281, 10.83513  , 0.392064])])

        mol_5 = Geometry(structure=[Atom(name="94", element= "C" , coords= [3.553798, 1.61359 , 4.8339 ]),
                Atom(name="95", element= "H" , coords= [4.326498, 2.37589 , 4.806883]),
                Atom(name="97", element= "Cl" , coords= [3.270833, 1.047785, 3.172041]),
                Atom(name="98", element= "Cl" , coords= [4.158172, 0.290949, 5.886787]),
                Atom(name="96", element= "Cl" , coords= [2.086952, 2.366203, 5.533697])])

        mol_6 = Geometry(structure=[Atom(name="99", element= "C" , coords= [-3.468257,  2.485975, -2.800501]),
                Atom(name="100", element= "H" , coords= [-3.066961,  1.783651, -2.070988]),
                Atom(name="101", element= "Cl" , coords= [-2.093225,  3.085446, -3.787699]),
                Atom(name="102", element= "Cl" , coords= [-4.654776,  1.619324, -3.825669]),
                Atom(name="103", element= "Cl" , coords= [-4.245191,  3.82967 , -1.913365])])

        mol_7 = Geometry(structure=[Atom(name="104", element= "C" , coords= [-6.669581, -3.00503 , -1.506201]),
                Atom(name="105", element= "H" , coords= [-7.211251, -2.785918, -0.590727]),
                Atom(name="107", element= "Cl" , coords= [-5.27109,  -4.037495, -1.074827]),
                Atom(name="106", element= "Cl" , coords= [-6.132572, -1.456226, -2.211052]),
                Atom(name="108", element= "Cl" , coords= [-7.791344, -3.866729, -2.602856])])
        ref = [mol_1, mol_2, mol_3, mol_4, mol_5, mol_6, mol_7]
        solv_list = g.detect_solvent(solvent="ClC(Cl)Cl")
        counter = 0
        for i, mol in enumerate(solv_list):
            if mol == ref[i]:
                counter += 1
        self.assertEqual(counter, len(ref))

def suite():
    suite = unittest.TestSuite()
    # suite.addTest(TestGeometry("test_vbur_lebedev"))
    # suite.addTest(TestGeometry("test_examine_constraints"))
    # suite.addTest(TestGeometry("test_detect_components"))
    suite.addTest(TestGeometry("test_canonical_rank"))
    # suite.addTest(TestGeometry("test_RMSD"))
    return suite


ONLYSOME = False

if __name__ == "__main__" and ONLYSOME:
    runner = unittest.TextTestRunner()
    runner.run(suite())
elif __name__ == "__main__":
    unittest.main()
