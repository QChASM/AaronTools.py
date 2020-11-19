#!/usr/bin/env python
import json
import os
import unittest

import numpy as np
from AaronTools.comp_output import CompOutput
from AaronTools.component import Component
from AaronTools.fileIO import Frequency
from AaronTools.geometry import Geometry
from AaronTools.json_extension import ATDecoder, ATEncoder
from AaronTools.substituent import Substituent
from AaronTools.test import TestWithTimer, prefix


class TestJSON(TestWithTimer):
    small_mol = os.path.join(prefix, "test_files", "benzene_1-NO2_4-Cl.xyz")
    COCH3 = os.path.join(prefix, "test_files", "COCH3.xyz")
    lig = os.path.join(prefix, "test_files", "ligands", "R-Quinox-tBu3.xyz")
    cat = os.path.join(prefix, "test_files", "catalysts", "tm_multi-lig.xyz")
    log = os.path.join(prefix, "test_files", "normal.log")

    def json_tester(self, ref, equality_function, as_iter=False, **kwargs):
        # test to json
        test = json.dumps(ref, cls=ATEncoder, indent=4)
        # test from json
        test = json.loads(test, cls=ATDecoder)
        if as_iter:
            for r, t in zip(ref, test):
                equality_function(r, t, **kwargs)
        else:
            equality_function(ref, test, **kwargs)

    def atom_equal(self, ref, test, skip=[]):
        for key, val in ref.__dict__.items():
            if key in skip:
                continue
            try:
                if key == "coords":
                    self.assertTrue(
                        np.linalg.norm(ref.coords - test.coords) < 10 ** -15
                    )
                elif key == "tags":
                    self.assertSetEqual(ref.tags, test.tags)
                elif key in ["connected", "constraint"]:
                    self.assertEqual(
                        len(ref.__dict__[key]), len(test.__dict__[key])
                    )
                    for a, b in zip(
                        sorted(ref.__dict__[key]), sorted(test.__dict__[key])
                    ):
                        if key == "constraint":
                            self.assertEqual(a[1], b[1])
                            a = a[0]
                            b = b[0]
                        self.atom_equal(a, b, skip=["connected", "constraint"])
                else:
                    self.assertEqual(ref.__dict__[key], test.__dict__[key])
            except AssertionError as err:
                print(ref, ref.__dict__[key])
                print(test, test.__dict__[key])
                raise AssertionError("(key={}) {}".format(key, err))

    def geom_equal(self, ref, test, skip=[]):
        if "name" not in skip:
            self.assertEqual(ref.name, test.name)
        else:
            skip.remove("name")
        if "comment" not in skip:
            self.assertEqual(ref.comment, test.comment)
        for a, b in zip(
            sorted(ref.atoms, key=lambda x: float(x.name)),
            sorted(test.atoms, key=lambda x: float(x.name)),
        ):
            self.atom_equal(a, b, skip)

    def sub_equal(self, ref, test):
        self.assertEqual(ref.name, test.name)
        for a, b in zip(ref.atoms, test.atoms):
            self.atom_equal(a, b)
        self.assertEqual(ref.end, test.end)
        self.assertEqual(ref.conf_num, test.conf_num)
        self.assertEqual(ref.conf_angle, test.conf_angle)

    def component_equal(self, ref, test):
        self.geom_equal(ref, test, skip=["name"])
        # need to sort substituents to make sure the order is the same
        # use the name of the first atom in the substituent
        self.assertEqual(len(ref.substituents), len(test.substituents))
        for r, t in zip(
            sorted(ref.substituents, key=lambda x: int(x.atoms[0].name)),
            sorted(test.substituents, key=lambda x: int(x.atoms[0].name)),
        ):
            self.geom_equal(r, t, skip=["comment"])
        self.assertEqual(len(ref.backbone), len(test.backbone))
        for r, t in zip(sorted(ref.backbone), sorted(test.backbone)):
            self.atom_equal(r, t)
        self.assertEqual(len(ref.key_atoms), len(test.key_atoms))
        for r, t in zip(sorted(ref.key_atoms), sorted(test.key_atoms)):
            self.atom_equal(r, t)

    def catalyst_equal(self, ref, test):
        # only one of them needs the comment re-parsed, but
        # it shouldn't matter if we do both
        ref.parse_comment()
        test.parse_comment()
        self.geom_equal(ref, test)
        if ref.center is None:
            ref.detect_components()
        if test.center is None:
            test.detect_components()
        for r, t in zip(ref.center, test.center):
            self.atom_equal(r, t)
        for r, t in zip(sorted(ref.components), sorted(test.components)):
            self.component_equal(r, t)

    def comp_out_equal(self, ref, test):
        keys = [
            "geometry",
            "opts",
            "frequency",
            "archive",
            "gradient",
            "E_ZPVE",
            "ZPVE",
            "energy",
            "enthalpy",
            "free_energy",
            "grimme_g",
            "charge",
            "multiplicity",
            "mass",
            "temperature",
            "rotational_temperature",
            "rotational_symmetry_number",
            "error",
            "error_msg",
            "finished",
        ]
        for key in keys:
            rval = ref.__dict__[key]
            tval = test.__dict__[key]
            if key == "geometry":
                self.geom_equal(rval, tval)
            elif key == "opts":
                if rval is None:
                    continue
                for r, t in zip(rval, tval):
                    self.geom_equal(r, t)
            elif key == "frequency":
                self.freq_equal(rval, tval)
            elif key == "gradient":
                self.assertDictEqual(rval, tval)
            elif key == "rotational_temperature":
                self.assertListEqual(rval, tval)
            else:
                self.assertEqual(rval, tval)

    def freq_equal(self, ref, test):
        for r, t in zip(ref.data, test.data):
            self.assertEqual(r.frequency, t.frequency)
            self.assertEqual(r.intensity, t.intensity)
            self.assertTrue(np.linalg.norm(r.vector - t.vector) < 10 ** -12)
        self.assertListEqual(
            ref.imaginary_frequencies, test.imaginary_frequencies
        )
        self.assertListEqual(ref.real_frequencies, test.real_frequencies)
        self.assertEqual(ref.is_TS, test.is_TS)
        self.assertEqual(
            len(ref.by_frequency.keys()), len(test.by_frequency.keys())
        )
        for key in ref.by_frequency.keys():
            self.assertTrue(key in test.by_frequency)
            rv = ref.by_frequency[key]
            tv = test.by_frequency[key]
            self.assertEqual(rv["intensity"], tv["intensity"])
            self.assertTrue(
                np.linalg.norm(rv["vector"] - tv["vector"]) < 10 ** -12
            )

    def test_atom(self):
        mol = Geometry(TestJSON.small_mol)
        for i, a in enumerate(mol.atoms):
            a.add_tag("test" + str(i))
        self.json_tester(
            mol.atoms,
            self.atom_equal,
            as_iter=True,
            skip=["connected", "constraint"],
        )

    def test_geometry(self):
        mol = Geometry(TestJSON.small_mol)
        self.json_tester(mol, self.geom_equal)

    def test_substituent(self):
        sub = Substituent("COCH3")
        self.json_tester(sub, self.sub_equal)

    def test_component(self):
        lig = Component(TestJSON.lig)
        self.json_tester(lig, self.component_equal)

    def test_catalyst(self):
        cat = Geometry(TestJSON.cat)
        cat.fix_comment()
        cat.conf_num = 4
        self.json_tester(cat, self.catalyst_equal)

    def test_comp_output(self):
        log = CompOutput(TestJSON.log)
        self.json_tester(log, self.comp_out_equal)

    def test_frequency(self):
        log = CompOutput(TestJSON.log)
        freq = log.frequency
        self.json_tester(freq, self.freq_equal)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestJSON("test_atom"))
    suite.addTest(TestJSON("test_geometry"))
    suite.addTest(TestJSON("test_substituent"))
    suite.addTest(TestJSON("test_component"))
    suite.addTest(TestJSON("test_comp_output"))
    suite.addTest(TestJSON("test_catalyst"))
    suite.addTest(TestJSON("test_frequency"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
