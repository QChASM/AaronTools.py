#! /usr/bin/env python3
import os
import random
import unittest

import numpy as np
from AaronTools.atoms import Atom, BondOrder
from AaronTools.geometry import Geometry
from AaronTools.test import TestWithTimer, prefix


class TestBondOrder(TestWithTimer):
    benzene_NO2_Cl = os.path.join(
        prefix, "test_files", "benzene_1-NO2_4-Cl.xyz"
    )

    def test_get(self):
        mol = Geometry(TestBondOrder.benzene_NO2_Cl)

        bond_data = BondOrder()
        for i, a in enumerate(mol.atoms):
            for c in mol.atoms[i + 1 :]:
                if c not in a.connected:
                    continue
                bo = bond_data.get(a, c)
                if a.element == "C" and c.element == "C":
                    # CC bonds should all be aromatic
                    self.assertEqual(bo, 1.5)
                elif (a.element == "N" and c.element == "O") or (
                    a.element == "O" and c.element == "N"
                ):
                    # ON bonds are 1.5
                    self.assertEqual(bo, 1.5)
                else:
                    # all other bonds are single
                    self.assertEqual(bo, 1)
        return


class TestAtoms(TestWithTimer):
    # Test constants
    benzene_NO2_Cl = os.path.join(
        prefix, "test_files", "benzene_1-NO2_4-Cl.xyz"
    )
    pentane = os.path.join(prefix, "test_files", "pentane.xyz")
    hexanol_diamine = os.path.join(prefix, "test_files", "6a2e5am1hex.xyz")

    # helper functions
    def read_xyz(self, fname):
        """just here to limit reliance on FileIO or Geometry"""
        mol = []
        with open(fname) as f:
            # split xyz file into info matrix
            tmp = f.read()
            tmp = tmp.split("\n")
            for line in tmp:
                line = line.strip().split()
                if len(line):
                    mol += [line]
            # remove atom number and comment line
            mol = mol[2:]
        atoms = []
        for i, a in enumerate(mol):
            a = Atom(element=a[0], coords=a[1:4], name=(i + 1), tags=i)
            atoms += [a]
        return atoms

    # test cases
    def test_store_atoms(self):
        atoms_read = self.read_xyz(TestAtoms.benzene_NO2_Cl)
        atoms_manual = []
        mol = []
        with open(TestAtoms.benzene_NO2_Cl) as f:
            # split xyz file into info matrix
            tmp = f.read()
            tmp = tmp.split("\n")
            for line in tmp:
                line = line.strip().split()
                if len(line):
                    mol += [line]
            # remove atom number and comment line
            mol = mol[2:]
        for i, line in enumerate(mol):
            a = Atom()
            a.element = line[0]
            a.coords = np.array(line[1:4], dtype=float)
            a.flag = False
            a.name = str(i + 1)
            a.tags = set([i])
            atoms_manual += [a]

        # test manual and **kwargs assignment match
        for a, b in zip(atoms_read, atoms_manual):
            self.assertEqual(a.element, b.element)
            self.assertTrue(np.linalg.norm(a.coords - b.coords) < 10 ** (-8))
            self.assertEqual(a.name, b.name)
            self.assertEqual(a.tags, b.tags)
            # should have automatic lookup of _radii
            self.assertTrue(a._radii is not None)

        # transform back into xyz format for verification
        atom_matrix = []
        for a in atoms_read:
            tmp = [str(a.element)]
            for c in a.coords:
                tmp += ["{:7.5f}".format(c)]
            atom_matrix += [tmp]
        for a, b in zip(mol, atom_matrix):
            a = " ".join(a)
            b = " ".join(b)
            self.assertEqual(a, b)

        return

    # utilities
    def test_float(self):
        atom_args = [
            {"element": "H", "coords": [0, 0, 0], "name": "1"},
            {"element": "H", "coords": [0, 0, 0], "name": "1.1"},
            {"element": "H", "coords": [0, 0, 0], "name": "1.1.1"},
        ]
        atom_floats = [1.0, 1.1, 1.1]
        for kwarg, ref in zip(atom_args, atom_floats):
            atom = Atom(**kwarg)
            self.assertTrue(float(atom) == ref)

    def test_less_than(self):
        atoms = Geometry(TestAtoms.benzene_NO2_Cl)
        atoms = atoms.atoms
        for atom in atoms:
            atom._rank = None
        # atom[11] has 3 connected atoms
        # atom[6] has 1 connected atom
        self.assertTrue(atoms[11] < atoms[6])
        # atom[12] is O
        # atom[10] is Cl
        self.assertTrue(atoms[12] < atoms[10])
        # atom[3] has 3 connected atoms: C, C, Cl
        # atom[2] has 3 connected atoms: C, C, H
        self.assertTrue(atoms[3] < atoms[2])

    def test_repr(self):
        atom = Atom("H", [0, 0, 0], True, "1", ["test"])
        s = repr(atom)
        self.assertEqual(
            s, "  H     0.00000000    0.00000000    0.00000000  -1    1"
        )

    def test_set_radii(self):
        atom = Atom("H", [0, 0, 0])
        self.assertEqual(atom._radii, 0.32)
        atom = Atom("X", [0, 0, 0])  # ghost atom
        self.assertEqual(atom._radii, 0.0)
        with self.assertRaises(ValueError):
            atom = Atom("NotAnElement", [0, 0, 0])
        atom = Atom()
        atom.element = "NotAnElement"
        with self.assertLogs(logger=Atom.LOG, level="WARNING"):
            atom._set_radii()

    def test_set_connectivity(self):
        atom = Atom("H", [0, 0, 0])
        self.assertEqual(atom._connectivity, 1)
        atom = Atom("C", [0, 0, 0])
        self.assertEqual(atom._connectivity, 4)

    def test_add_tag(self):
        mol = self.read_xyz(TestAtoms.benzene_NO2_Cl)
        for atom in mol:
            new_tags = [random.random() for i in range(10)]
            old_tags = atom.tags
            atom.add_tag(new_tags[0])
            atom.add_tag(new_tags[1], new_tags[2], new_tags[3], new_tags[4])
            atom.add_tag(new_tags[5:])
            self.assertSequenceEqual(
                sorted(old_tags.union(new_tags)), sorted(atom.tags)
            )

    def test_get_invariant(self):
        pentane = Geometry(TestAtoms.pentane)
        mol = Geometry(TestAtoms.hexanol_diamine)

        s = ""
        for a in pentane.atoms:
            if a.element == "H":
                continue
            s += a.get_invariant() + " "
        self.assertEqual(s, "10100063 20200062 20200062 20200062 10100063 ")

        s = ""
        for a in mol.atoms:
            if a.element == "H":
                continue
            s += a.get_invariant() + " "
        ans = "30300061 20200062 20200062 30300061 20200062 20200062 20200062 20200062 10100081 10100063 10100072 10100072 "
        self.assertEqual(s, ans)

    # measurement
    def test_is_connected(self):
        atoms = self.read_xyz(TestAtoms.benzene_NO2_Cl)
        conn_valid = [
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
        for i, a in enumerate(atoms):
            for j, b in enumerate(atoms[i + 1 :]):
                if a.is_connected(b):
                    a.connected.add(b)
                    b.connected.add(a)
            conn_try = sorted([int(c.name) for c in a.connected])
            conn_try = ",".join([str(c) for c in conn_try])
            self.assertEqual(conn_try, conn_valid[i])
        return

    def test_bond(self):
        a1 = Atom(element="H", coords=[0, 0, 0])
        a2 = Atom(element="H", coords=[1, 0, 0])
        self.assertTrue(np.linalg.norm(a1.bond(a2) - [1, 0, 0]) == 0)

    def test_dist(self):
        a1 = Atom(element="H", coords=[0, 0, 0])
        a2 = Atom(element="H", coords=[1, 0, 0])
        self.assertTrue(a1.dist(a2) == 1)

    def test_angle(self):
        mol = self.read_xyz(TestAtoms.benzene_NO2_Cl)
        angle = mol[11].angle(mol[12], mol[13])
        self.assertTrue(abs(np.rad2deg(angle) - 124.753) < 10 ** (-3))


if __name__ == "__main__":
    unittest.main()
