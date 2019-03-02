#!/usr/bin/env python

import unittest
import json
import numpy as np

from AaronTools.test import prefix, TestWithTimer
from AaronTools.json_extension import JSONEncoder, JSONDecoder
from AaronTools.geometry import Geometry
from AaronTools.substituent import Substituent
from AaronTools.component import Component
from AaronTools.catalyst import Catalyst
from AaronTools.comp_output import CompOutput


class TestJSON(TestWithTimer):
    small_mol = prefix + "test_files/benzene_1-NO2_4-Cl.xyz"
    COCH3 = prefix + 'test_files/COCH3.xyz'
    lig = prefix + 'test_files/ligands/R-Quinox-tBu3.xyz'
    cat = prefix + 'test_files/catalysts/tm_multi-lig.xyz'
    log = prefix + 'test_files/normal.log'

    def json_tester(self, obj, ref_json, equality_function, as_iter=False, **kwargs):
        # test to json
        test = json.dumps(obj, cls=JSONEncoder)
        with open(ref_json) as f:
            self.assertEqual(test, f.read())
        # test from json
        with open(ref_json) as f:
            test = json.load(f, cls=JSONDecoder)
        if as_iter:
            for r, t in zip(obj, test):
                equality_function(r, t, **kwargs)
        else:
            equality_function(obj, test, **kwargs)

    def atom_equal(self, ref, test, skip=[]):
        for key, val in ref.__dict__.items():
            if key in skip:
                continue
            if key == 'coords':
                self.assertTrue(np.linalg.norm(
                    ref.coords - test.coords) < 10**-15)
            elif key in ['tags', 'connected', 'constraint']:
                self.assertSetEqual(ref.tags, test.tags)
            else:
                self.assertEqual(ref.__dict__[key], test.__dict__[key])

    def geom_equal(self, ref, test):
        self.assertEqual(ref.name, test.name)
        self.assertEqual(ref.comment, test.comment)
        for a, b in zip(ref.atoms, test.atoms):
            self.atom_equal(a, b)

    def sub_equal(self, ref, test):
        self.assertEqual(ref.name, test.name)
        for a, b in zip(ref.atoms, test.atoms):
            self.atom_equal(a, b)
        self.assertEqual(ref.end, test.end)
        self.assertEqual(ref.conf_num, test.conf_num)
        self.assertEqual(ref.conf_angle, test.conf_angle)

    def component_equal(self, ref, test):
        self.geom_equal(ref, test)
        self.assertEqual(len(ref.substituents), len(test.substituents))
        for r, t in zip(ref.substituents, test.substituents):
            self.geom_equal(r, t)
        self.assertEqual(len(ref.backbone), len(test.backbone))
        for r, t in zip(ref.backbone, test.backbone):
            self.atom_equal(r, t)
        self.assertEqual(len(ref.key_atoms), len(test.key_atoms))
        for r, t in zip(ref.key_atoms, test.key_atoms):
            self.atom_equal(r, t)

    def catalyst_equal(self, ref, test):
        self.geom_equal(ref, test)
        for r, t in zip(ref.center, test.center):
            self.atom_equal(r, t)
        for key in ref.components:
            for r, t in zip(ref.components[key], test.components[key]):
                self.component_equal(r, t)
        self.assertEqual(ref.conf_num, test.conf_num)

    def comp_out_equal(self, ref, test):
        keys = [
            'geometry', 'opts', 'frequency', 'archive',
            'gradient', 'E_ZPVE', 'ZPVE',
            'energy', 'enthalpy', 'free_energy', 'grimme_g',
            'charge', 'multiplicity', 'mass', 'temperature',
            'rotational_temperature', 'rotational_symmetry_number',
            'error', 'error_msg', 'finished'
        ]
        for key in keys:
            rval = ref.__dict__[key]
            tval = ref.__dict__[key]
            if key == 'geometry':
                self.geom_equal(rval, tval)
            elif key == 'opts':
                for r, t in zip(rval, tval):
                    self.geom_equal(r, t)
            elif key == 'frequency':
                self.freq_equal(rval, tval)
            elif key == 'gradient':
                self.assertDictEqual(rval, tval)
            elif key == 'rotational_temperature':
                self.assertListEqual(rval, tval)
            else:
                self.assertEqual(rval, tval)

    def freq_equal(self, ref, test):
        for r, t in zip(ref.data, test.data):
            self.assertEqual(r.frequency, t.frequency)
            self.assertEqual(r.intensity, t.intensity)
            self.assertDictEqual(r.vector, t.vector)
        self.assertListEqual(ref.imaginary_frequencies,
                             test.imaginary_frequencies)
        self.assertListEqual(ref.real_frequencies, test.real_frequencies)
        self.assertDictEqual(ref.by_frequency, test.by_frequency)
        self.assertEqual(ref.is_TS, test.is_TS)

    def test_atom(self):
        mol = Geometry(TestJSON.small_mol)
        for i, a in enumerate(mol.atoms):
            a.add_tag('test' + str(i))
        ref_file = prefix + 'ref_files/atom.json'
        self.json_tester(mol.atoms, ref_file, self.atom_equal,
                         as_iter=True,
                         skip=['connected', 'constraint', '_rank'])

    def test_geometry(self):
        mol = Geometry(TestJSON.small_mol)
        ref_file = prefix + 'ref_files/geometry.json'
        self.json_tester(mol, ref_file, self.geom_equal)

    def test_substituent(self):
        sub = Substituent(TestJSON.COCH3)
        ref_file = prefix + 'ref_files/sub.json'
        self.json_tester(sub, ref_file, self.sub_equal)

    def test_component(self):
        lig = Component(TestJSON.lig)
        ref_file = prefix + 'ref_files/lig.json'
        self.json_tester(lig, ref_file, self.component_equal)

    def test_catalyst(self):
        cat = Catalyst(TestJSON.cat)
        cat.conf_num = 4
        ref_file = prefix + 'ref_files/cat.json'
        self.json_tester(cat, ref_file, self.catalyst_equal)

    def test_comp_output(self):
        log = CompOutput(TestJSON.log)
        ref_file = prefix + 'ref_files/log.json'
        self.json_tester(log, ref_file, self.comp_out_equal)

    def test_frequency(self):
        log = CompOutput(TestJSON.log)
        freq = log.frequency
        ref_file = prefix + 'ref_files/freq.json'
        self.json_tester(freq, ref_file, self.freq_equal)


if __name__ == '__main__':
    unittest.main()
