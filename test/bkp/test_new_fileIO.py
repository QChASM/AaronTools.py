#!/usr/bin/env python3
import unittest
import numpy as np

from AaronTools.new_fileIO import FileReader
from AaronTools.test import prefix, TestWithTimer


class TestFileReader(TestWithTimer):
    small_mol = prefix + "test_files/benzene_1-NO2_4-Cl.xyz"

    def xyz_matrix(self, fname):
        rv = []
        with open(fname) as f:
            rv += [f.readline().strip()]
            rv += [f.readline().strip()]
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                rv += [line.split()]
        return rv

    def validate_atoms(self, ref, test, skip=[]):
        for r, t in zip(ref.atoms, test.atoms):
            if r.element != t.element:
                return False
            if np.linalg.norm(r.coords - t.coords) > 10**-6:
                return False
            if 'name' not in skip and r.name != t.name:
                return False
            if 'flag' not in skip and r.flag != t.flag:
                return False
            if 'tags' not in skip:
                if len(r.tags - t.tags) > 0:
                    return False
            return True

    def test_read_xyz_structure(self):
        ref = self.xyz_matrix(TestFileReader.small_mol)
        mol = FileReader(TestFileReader.small_mol)
        for i, line in enumerate(ref):
            if i == 0:
                # check number of atoms
                self.assertEqual(len(mol.atoms), int(line))
                continue
            if i == 1:
                # check comment
                self.assertEqual(mol.comment, line)
                continue
            # check element and coordinates
            n = i - 2  # atoms start at line 2
            a = mol.atoms[n]
            self.assertEqual(a.element, line[0])
            dist = a.coords - np.array(line[1:], dtype=float)
            dist = np.linalg.norm(dist)
            self.assertTrue(dist < 10**(-8))
            # check name and tags
            self.assertEqual(str(n+1), a.name)
            self.assertTrue(len(a.tags) == 0)
        return

    def test_read_log_structure(self):
        ref = FileReader("ref_files/file_io_normal.xyz")
        test = FileReader(prefix + "test_files/normal.log")
        self.assertTrue(self.validate_atoms(ref, test))

        ref = FileReader("ref_files/file_io_error.xyz")
        test = FileReader(prefix + "test_files/error.log")
        self.assertTrue(self.validate_atoms(ref, test))

        ref = FileReader("ref_files/file_io_died.xyz")
        test = FileReader(prefix + "test_files/died.log")
        self.assertTrue(self.validate_atoms(ref, test))


if __name__ == '__main__':
    unittest.main()
