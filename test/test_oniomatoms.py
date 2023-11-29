#! /usr/bin/env python3
import unittest
import random
import numpy as np
import os

from AaronTools.atoms import Atom, BondOrder
from AaronTools.geometry import Geometry
from AaronTools.test import prefix, TestWithTimer
from AaronTools.oniomatoms import OniomAtom, ChargeSum, OniomSanity

class TestBondOrder(TestWithTimer):
    small_mol = os.path.join(prefix, "test_files", "malonate-3_waters.xyz")

    def test_get(self):
        mol = Geometry(TestBondOrder.small_mol)

        bond_data = BondOrder()
        for i, a in enumerate(mol.atoms):
            for j, c in enumerate(mol.atoms):
                if j != i:
                    if c not in a.connected:
                        continue
                    bo = bond_data.get(a, c)
                    if a.element == "C" and c.element == "C":
                        # C-C bonds should all be single
                        self.assertEqual(bo, 1)
                    elif (a.element == "C" and c.element == "O" and "H" in c.connected) \
                            or (c.element == "C" and a.element == "O" and "H" in a.connected):
                        # C-OH bonds are single
                        self.assertEqual(bo, 1)
                    elif (a.element == "C" and c. element == "O" and "H" not in c.connected) \
				            or (c.element == "C" and a.element == "O" and "H" not in a.connected):
                        for k, d in enumerate(mol.atoms):
                            if k != i and k != j:
                                if (a.element =="C" and d in a.connected and d == "O") \
                                      or (c.element == "C" and d in c.connected and d == "O"):
                                    if "H" in d.connected:
                                        # Carbonyl of a carboxylic acid is double
                                        self.assertEqual(bo, 2)
                                    else:
                                        # Carbonyl of a carboxylate is 1.5
                                        self.assertEqual(bo, 1.5)
                    else:
                        # all other bonds are single
                        self.assertEqual(bo, 1)
        return


class TestOniomAtoms(TestWithTimer):
    # Test constants
    small_mol = os.path.join(prefix, "test_files", "malonate-3_waters.xyz")

    # helper functions
    def read_xyz(self, fname):
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
            a = OniomAtom(element=a[0], coords=a[1:4], name=(i + 1), tags=i, layer=a[4], atomtype=a[5], charge=a[6])
            atoms += [a]
        return atoms

    # test cases
    def test_store_atoms(self):
        atoms_read = self.read_xyz(TestOniomAtoms.small_mol)
        atoms_manual = []
        mol = []
        with open(TestOniomAtoms.small_mol) as f:
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
            a = OniomAtom()
            a.element = line[0]
            a.coords = np.array(line[1:4], dtype=float)
            a.flag = False
            a.name = str(i + 1)
            a.tags.add(i)
            a.layer = line[4]
            a.atomtype = line[5]
            a.charge = line[6]
            atoms_manual += [a]
        # test manual and **kwargs assignment match
        for a, b in zip(atoms_read, atoms_manual):
            self.assertEqual(a.element, b.element)
            self.assertTrue(np.linalg.norm(a.coords - b.coords) < 10**(-8))
            self.assertEqual(a.name, b.name)
            self.assertEqual(a.tags, b.tags)
            self.assertEqual(a.layer, b.layer)
            # should have automatic lookup of _radii
            self.assertTrue(a._radii is not None)
        # transform back into xyz format for verification
        atom_matrix = []
        for a in atoms_read:
            tmp = [str(a.element)]
            for c in a.coords:
                tmp += ["{}".format(c)]
            tmp += [str(a.layer)]
            tmp += [str(a.atomtype)]
            tmp += [str(a.charge)]
            atom_matrix += [tmp]
        for a, b in zip(mol, atom_matrix):
            a = ' '.join(a)
            b = ' '.join(b)
            self.assertEqual(a, b)

        return

    # utilities
    def test_float(self):
        atom_args = [
            {'element': 'H', 'coords': [0, 0, 0], 'name': '1'},
            {'element': 'H', 'coords': [0, 0, 0], 'name': '1.1'},
            {'element': 'H', 'coords': [0, 0, 0], 'name': '1.1.1'}
        ]
        atom_floats = [1., 1.1, 1.1]
        for kwarg, ref in zip(atom_args, atom_floats):
            atom = OniomAtom(**kwarg)
            self.assertTrue(float(atom) == ref)

    def test_less_than(self):
        atoms = Geometry(TestOniomAtoms.small_mol)
        atoms = atoms.atoms
        self.assertTrue(atoms[2] < atoms[3])
        self.assertTrue(atoms[4] < atoms[2])
        self.assertTrue(atoms[5] < atoms[3])

    def test_greater_than(self):
        atoms = Geometry(TestOniomAtoms.small_mol)
        atoms = atoms.atoms
        self.assertTrue(atoms[0] > atoms[10])
        self.assertTrue(atoms[0] > atoms[11])
        self.assertTrue(atoms[0] > atoms[12])
        self.assertTrue(atoms[0] > atoms[13])
        self.assertTrue(atoms[0] > atoms[14])
        self.assertTrue(atoms[0] > atoms[15])
        self.assertTrue(atoms[0] > atoms[16])
        self.assertTrue(atoms[0] > atoms[17])
        self.assertTrue(atoms[0] > atoms[18])

    def test_add_tag(self):
        mol = self.read_xyz(TestOniomAtoms.small_mol)
        for atom in mol:
            new_tags = [random.random() for i in range(10)]
            old_tags = atom.tags
            atom.add_tag(new_tags[0])
            atom.add_tag(new_tags[1], new_tags[2], new_tags[3], new_tags[4])
            atom.add_tag(new_tags[5:])
            self.assertSequenceEqual(
                sorted(old_tags.union(new_tags)), sorted(atom.tags))

    def test_get_invariant(self):
        pentane = Geometry(os.path.join(prefix, "test_files", "pentane.xyz"))
        mol = Geometry(os.path.join(prefix, "test_files", "6a2e5am1hex.xyz"))

        s = ''
        for a in pentane.atoms:
            if a.element == 'H':
                continue
            s += a.get_invariant() + " "
        self.assertEqual(s, "10100063 20200062 20200062 20200062 10100063 ")

        s = ''
        for a in mol.atoms:
            if a.element == 'H':
                continue
            s += a.get_invariant() + " "
        ans = "30300061 20200062 20200062 30300061 20200062 20200062 20200062 20200062 10100081 10100063 10100072 10100072 "
        self.assertEqual(s, ans)

    # measurement

    def test_is_connected(self):
        atoms = self.read_xyz(TestOniomAtoms.small_mol)
        conn_valid = ['2,3,4',
                      '1',
                      '1,5,8,9',
                      '1,10',
                      '3,6,7',
                      '5',
                      '5',
                      '3',
                      '3',
                      '4',
                      '12',
                      '11,13',
                      '12',
                      '15',
                      '14,16',
                      '15',
                      '18',
                      '17,19',
                      '18']
        for i, a in enumerate(atoms):
            for j, b in enumerate(atoms[i+1:]):
                if a.is_connected(b):
                    a.connected.add(b)
                    b.connected.add(a)
            conn_try = sorted([int(c.name) for c in a.connected])
            conn_try = ','.join([str(c) for c in conn_try])
            self.assertEqual(conn_try, conn_valid[i])
        return

    def test_bond(self):
        a1 = Atom(element='H', coords=[0, 0, 0])
        a2 = Atom(element='H', coords=[1, 0, 0])
        self.assertTrue(np.linalg.norm(a1.bond(a2) - [1, 0, 0]) == 0)

    def test_dist(self):
        a1 = Atom(element='H', coords=[0, 0, 0])
        a2 = Atom(element='H', coords=[1, 0, 0])
        self.assertTrue(a1.dist(a2) == 1)

    def test_angle(self):
        mol = self.read_xyz(TestOniomAtoms.small_mol)
        angle = mol[17].angle(mol[16], mol[18])
        self.assertTrue(abs(np.rad2deg(angle) - 101.1) < 10**(-1))

class TestChargeSum(TestWithTimer): 
    atoms = Geometry(TestOniomAtoms.small_mol)
    tot_charge = ChargeSum().get(atoms.atoms)
    diff = abs(tot_charge-(-1))
    def test_charge_sum(self):
        self.assertTrue(self.diff < 0.000001)

class TestOniomSanity(TestWithTimer):
    a = TestOniomAtoms()
    atomlist = a.read_xyz(TestOniomAtoms.small_mol)
 
    def test_check_charges(self):
        self.assertTrue(OniomSanity.check_charges(self.atomlist[:],-1))

    def test_check_types(self):
        self.assertTrue(OniomSanity.check_types(self.atomlist[:]))

    def test_check_layers(self):
        self.assertTrue(OniomSanity.check_layers(self.atomlist))

if __name__ == "__main__":
    unittest.main()
