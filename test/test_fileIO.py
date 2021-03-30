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

        ref = """#n PBE1PBE/gen freq=(temperature=298.15,HPModes,NoRaman) opt=(VeryTight) EmpiricalDispersion=(GD3BJ) scrf=(SMD,solvent=dichloromethane)

comment line 1
comment line 2

0 1
C     -1.97696    -2.32718     0.00126
C     -2.36814    -1.29554     0.85518
C     -1.67136    -0.08735     0.85440
C     -0.58210     0.08919     0.00026
C     -0.19077    -0.94241    -0.85309
C     -0.88848    -2.15056    -0.85289
H     -3.22679    -1.43483     1.52790
H     -1.98002     0.72606     1.52699
H      0.66766    -0.80358    -1.52636
H     -0.57992    -2.96360    -1.52585
Cl     0.29699     1.61392    -0.00037
N     -2.73689    -3.64357     0.00188
O     -2.07823    -4.68230     0.00289
O     -3.96579    -3.59263     0.00134

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
C    -1.97696  -2.32718   0.00126
C    -2.36814  -1.29554   0.85518
C    -1.67136  -0.08735   0.85440
C    -0.58210   0.08919   0.00026
C    -0.19077  -0.94241  -0.85309
C    -0.88848  -2.15056  -0.85289
H    -3.22679  -1.43483   1.52790
H    -1.98002   0.72606   1.52699
H     0.66766  -0.80358  -1.52636
H    -0.57992  -2.96360  -1.52585
Cl    0.29699   1.61392  -0.00037
N    -2.73689  -3.64357   0.00188
O    -2.07823  -4.68230   0.00289
O    -3.96579  -3.59263   0.00134
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
     C       -1.97696       -2.32718        0.00126
     C       -2.36814       -1.29554        0.85518
     C       -1.67136       -0.08735        0.85440
     C       -0.58210        0.08919        0.00026
     C       -0.19077       -0.94241       -0.85309
     C       -0.88848       -2.15056       -0.85289
     H       -3.22679       -1.43483        1.52790
     H       -1.98002        0.72606        1.52699
     H        0.66766       -0.80358       -1.52636
     H       -0.57992       -2.96360       -1.52585
    Cl        0.29699        1.61392       -0.00037
     N       -2.73689       -3.64357        0.00188
     O       -2.07823       -4.68230        0.00289
     O       -3.96579       -3.59263        0.00134
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

        for line1, line2 in zip(test.splitlines(), ref.splitlines()):
            self.assertEqual(line1.strip(), line2.strip())

    def test_read_crest(self):
        fname = os.path.join(prefix, "test_files/good.crest")
        res = FileReader(fname, conf_name=False)
        self.assertDictEqual(
            res.other,
            {
                "best_energy": -79.18948,
                "temperature": 298.15,
                "energy": 0.172,
                "entropy": 0.006899,
                "free_energy": -2.057,
                "best_pop": 70.368,
                "finished": True,
                "error": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
