#!/usr/bin/env python3
import os
import unittest

import numpy as np
from AaronTools.fileIO import FileReader, FileWriter
from AaronTools.geometry import Geometry
from AaronTools.test import TestWithTimer, prefix
from AaronTools.theory import *


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
        ref = FileReader(os.path.join(prefix, "ref_files", "orca_geom.xyz"))
        test = FileReader(os.path.join(prefix, "test_files", "orca_geom.out"))
        self.assertTrue(self.validate_atoms(ref, test))

    def test_read_psi4_dat_structure(self):
        ref = FileReader(os.path.join(prefix, "ref_files", "psi4_geom.xyz"))
        test = FileReader((self.psi4_output_file, "dat", None))
        self.assertTrue(self.validate_atoms(ref, test))

    def test_read_log_structure(self):
        ref = FileReader(
            os.path.join(prefix, "ref_files", "file_io_normal.xyz")
        )
        test = FileReader(os.path.join(prefix, "test_files", "normal.log"))
        self.assertTrue(self.validate_atoms(ref, test))

        ref = FileReader(
            os.path.join(prefix, "ref_files", "file_io_error.xyz")
        )
        test = FileReader(os.path.join(prefix, "test_files", "error.log"))
        self.assertTrue(self.validate_atoms(ref, test))

        ref = FileReader(os.path.join(prefix, "ref_files", "file_io_died.xyz"))
        test = FileReader(os.path.join(prefix, "test_files", "died.log"))
        self.assertTrue(self.validate_atoms(ref, test))

    def test_read_com_info(self):
        """testing if we can read route info"""
        ref1 = {
            "method": "B3LYP/aug-cc-pVDZ",
            "temperature": "298.15",
            "solvent": "1,1,1-TriChloroEthane",
            "solvent_model": "PCM",
            "emp_dispersion": "GD3",
            "grid": "SuperFineGrid",
            "comment": "testing 1 2 3\ntesting 1 2 3",
            "charge": 0,
            "multiplicity": 1,
        }

        test = FileReader(self.com_file1, just_geom=False)
        self.assertEqual(test.other, ref1)

        ref2 = {
            "method": "B3LYP/aug-cc-pVDZ",
            "temperature": "298.15",
            "solvent": "1,1,1-TriChloroEthane",
            "solvent_model": "PCM",
            "emp_dispersion": "GD3",
            "grid": "ultrafinegrid",
            "comment": "testing 1 2 3\ntesting 1 2 3",
            "charge": 0,
            "multiplicity": 1,
        }

        test = FileReader(self.com_file2, just_geom=False)
        self.assertEqual(test.other, ref2)

    def test_write_com(self):
        """write gaussian input file"""
        # this compares the exact string, not things like RMSD or dictionaries
        # if it fails, someone may have added a column of whitespace or something
        geom = Geometry(self.small_mol)

        ref = """#n PBE1PBE/gen freq=(temperature=298.15,HPModes,NoRaman) opt=VeryTight EmpiricalDispersion=GD3BJ scrf=(SMD,solvent=dichloromethane)

comment line 1
comment line 2

0 1
C      -1.976960    -2.327180     0.001260
C      -2.368140    -1.295540     0.855180
C      -1.671360    -0.087350     0.854400
C      -0.582100     0.089190     0.000260
C      -0.190770    -0.942410    -0.853090
C      -0.888480    -2.150560    -0.852890
H      -3.226790    -1.434830     1.527900
H      -1.980020     0.726060     1.526990
H       0.667660    -0.803580    -1.526360
H      -0.579920    -2.963600    -1.525850
Cl      0.296990     1.613920    -0.000370
N      -2.736890    -3.643570     0.001880
O      -2.078230    -4.682300     0.002890
O      -3.965790    -3.592630     0.001340

H 0
def2SVP
****
C 0
def2TZVP
****



"""

        theory = Theory(
            charge=0,
            multiplicity=1,
            method="PBE0",
            basis=BasisSet(
                [Basis("def2-SVP", ["H"]), Basis("def2-TZVP", ["C"])]
            ),
            empirical_dispersion=EmpiricalDispersion("D3BJ"),
            solvent=ImplicitSolvent("SMD", "dichloromethane"),
            job_type=[FrequencyJob(), OptimizationJob()],
        )

        kw_dict = {
            GAUSSIAN_ROUTE: {
                "opt": ["VeryTight"],
                "freq": ["HPModes", "NoRaman"],
            },
            GAUSSIAN_COMMENT: ["comment line 1", "comment line 2"],
        }

        test = FileWriter.write_com(
            geom, theory=theory, outfile=False, **kw_dict
        )

        for line1, line2 in zip(test.splitlines(), ref.splitlines()):
            self.assertEqual(line1.strip(), line2.strip())

    def test_write_inp(self):
        """write orca input file"""
        # like gaussian input files, this compares exact output

        geom = Geometry(self.small_mol)

        ref = """#comment line 1
#comment line 2
! PBE0 D3BJ CPCM(dichloromethane) def2-SVP Freq Opt
%cpcm
    smd    true
end
%basis
    newGTO            C  "def2-TZVP" end
end
%freq
    Temp    298.15
end

*xyz 0 1
C    -1.976960  -2.327180   0.001260
C    -2.368140  -1.295540   0.855180
C    -1.671360  -0.087350   0.854400
C    -0.582100   0.089190   0.000260
C    -0.190770  -0.942410  -0.853090
C    -0.888480  -2.150560  -0.852890
H    -3.226790  -1.434830   1.527900
H    -1.980020   0.726060   1.526990
H     0.667660  -0.803580  -1.526360
H    -0.579920  -2.963600  -1.525850
Cl    0.296990   1.613920  -0.000370
N    -2.736890  -3.643570   0.001880
O    -2.078230  -4.682300   0.002890
O    -3.965790  -3.592630   0.001340
*

"""

        theory = Theory(
            charge=0,
            multiplicity=1,
            method="PBE0",
            basis=BasisSet(
                [Basis("def2-SVP", ["H"]), Basis("def2-TZVP", ["C"])]
            ),
            empirical_dispersion=EmpiricalDispersion("D3BJ"),
            solvent=ImplicitSolvent("SMD", "dichloromethane"),
            job_type=[FrequencyJob(), OptimizationJob()],
        )

        kw_dict = {ORCA_COMMENT: ["comment line 1", "comment line 2"]}

        test = FileWriter.write_inp(
            geom, theory=theory, outfile=False, **kw_dict
        )

        for line1, line2 in zip(test.splitlines(), ref.splitlines()):
            self.assertEqual(line1.strip(), line2.strip())

    def test_write_in(self):
        """write psi4 input file"""
        # like gaussian input files, this compares exact output

        geom = Geometry(self.small_mol)

        ref = """#comment line 1
#comment line 2
basis {
    assign    def2-SVP
    assign C  def2-TZVP
}

molecule {
 0 1
C    -1.976960  -2.327180   0.001260
C    -2.368140  -1.295540   0.855180
C    -1.671360  -0.087350   0.854400
C    -0.582100   0.089190   0.000260
C    -0.190770  -0.942410  -0.853090
C    -0.888480  -2.150560  -0.852890
H    -3.226790  -1.434830   1.527900
H    -1.980020   0.726060   1.526990
H     0.667660  -0.803580  -1.526360
H    -0.579920  -2.963600  -1.525850
Cl    0.296990   1.613920  -0.000370
N    -2.736890  -3.643570   0.001880
O    -2.078230  -4.682300   0.002890
O    -3.965790  -3.592630   0.001340
}

set {
    T                       298.15
}

nrg = frequencies('PBE0-d3bj')
nrg, wfn = optimize('PBE0-d3bj', return_wfn=True)
"""

        theory = Theory(
            charge=0,
            multiplicity=1,
            method="PBE0",
            basis=BasisSet(
                [Basis("def2-SVP", ["H"]), Basis("def2-TZVP", ["C"])]
            ),
            empirical_dispersion=EmpiricalDispersion("D3BJ"),
            job_type=[FrequencyJob(), OptimizationJob()],
        )

        kw_dict = {
            PSI4_COMMENT: ["comment line 1", "comment line 2"],
            PSI4_JOB: {"optimize": ["return_wfn=True"]},
        }

        test = FileWriter.write_in(
            geom, theory=theory, outfile=False, **kw_dict
        )

        self.assertEqual(test.splitlines(), ref.splitlines())

        for line1, line2 in zip(test.splitlines(), ref.splitlines()):
            self.assertEqual(line1.strip(), line2.strip())


if __name__ == "__main__":
    unittest.main()
