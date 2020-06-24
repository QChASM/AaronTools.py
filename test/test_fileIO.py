#!/usr/bin/env python3
import os
import unittest

import numpy as np

from AaronTools.fileIO import FileReader
from AaronTools.test import TestWithTimer, prefix


class TestFileReader(TestWithTimer):
    small_mol = os.path.join(prefix, "test_files", "benzene_1-NO2_4-Cl.xyz")
    com_file1 = os.path.join(prefix, "test_files", "test-route.com")
    com_file2 = os.path.join(prefix, "test_files", "test-route-2.com")
    psi4_output_file = os.path.join(prefix, "test_files", "psi4-test.out")

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
            if np.linalg.norm(r.coords - t.coords) > 10 ** -5:
                return False
            if "name" not in skip and r.name != t.name:
                return False
            if "flag" not in skip and r.flag != t.flag:
                return False
            if "tags" not in skip:
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
            self.assertTrue(dist < 10 ** (-8))
            # check name and tags
            self.assertEqual(str(n + 1), a.name)
            self.assertTrue(len(a.tags) == 0)

    def test_read_orca_out_structure(self):
        ref = FileReader("ref_files/orca_geom.xyz")
        test = FileReader(os.path.join(prefix, "test_files", "orca_geom.out"))
        self.assertTrue(self.validate_atoms(ref, test))

    def test_read_psi4_dat_structure(self):
        ref = FileReader(os.path.join(prefix, "ref_files", "psi4_geom.xyz"))
        test = FileReader((self.psi4_output_file, 'dat', None))
        self.assertTrue(self.validate_atoms(ref, test))

    def test_read_log_structure(self):
        ref = FileReader("ref_files/file_io_normal.xyz")
        test = FileReader(os.path.join(prefix, "test_files", "normal.log"))
        self.assertTrue(self.validate_atoms(ref, test))

        ref = FileReader("ref_files/file_io_error.xyz")
        test = FileReader(os.path.join(prefix, "test_files", "error.log"))
        self.assertTrue(self.validate_atoms(ref, test))

        ref = FileReader("ref_files/file_io_died.xyz")
        test = FileReader(os.path.join(prefix, "test_files", "died.log"))
        self.assertTrue(self.validate_atoms(ref, test))

    def test_read_com_info(self):
        """testing if we can read route info"""
        ref1 = {'method': 'B3LYP/aug-cc-pVDZ', \
                'temperature': '298.15', \
                'solvent': '1,1,1-TriChloroEthane', \
                'solvent_model': 'PCM', \
                'emp_dispersion': 'GD3', \
                'grid': 'SuperFineGrid', \
                'comment': 'testing 1 2 3\ntesting 1 2 3', \
                'charge': 0, \
                'multiplicity': 1}

        test = FileReader(self.com_file1, just_geom=False)
        self.assertEqual(test.other, ref1)

        ref2 = {'method': 'B3LYP/aug-cc-pVDZ', \
                'temperature': '298.15', \
                'solvent': '1,1,1-TriChloroEthane', \
                'solvent_model': 'PCM', \
                'emp_dispersion': 'GD3', \
                'grid': 'ultrafinegrid', \
                'comment': 'testing 1 2 3\ntesting 1 2 3', \
                'charge': 0, \
                'multiplicity': 1}

        test = FileReader(self.com_file2, just_geom=False)
        self.assertEqual(test.other, ref2)

if __name__ == "__main__":
    unittest.main()
