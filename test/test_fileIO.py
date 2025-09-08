#!/usr/bin/env python3
import os
import unittest

import numpy as np
from AaronTools.fileIO import FileReader, FileWriter
from AaronTools.geometry import Geometry
from AaronTools.test import TestWithTimer, prefix
from AaronTools.theory import *


class TestFileIO(TestWithTimer):
    small_mol = os.path.join(prefix, "test_files", "benzene_1-NO2_4-Cl.xyz")
    com_file1 = os.path.join(prefix, "test_files", "test-route.com")
    com_file2 = os.path.join(prefix, "test_files", "test-route-2.com")
    psi4_output_file = os.path.join(prefix, "test_files", "psi4-test.out")
    orca_orbit_file = os.path.join(prefix, "test_files", "pople.out")
    gaussian_mp2_file = os.path.join(prefix, "test_files", "mp2.log")
    gaussian_mp4_file = os.path.join(prefix, "test_files", "mp4.log")
    gaussian_ccsd_file = os.path.join(prefix, "test_files", "ccsd.log")
    gaussian_ccsd_t_file = os.path.join(prefix, "test_files", "ccsd_t.log")
    gaussian_dhdft_file = os.path.join(prefix, "test_files", "b2plypd3.log")
    ts = os.path.join(prefix, "test_files", "claisen_ts.xyz")
    oniom_com = os.path.join(prefix, "test_files", "pd_complex_2.com")
    oniom_xyz = os.path.join(prefix, "test_files", "pd_complex_2.xyz")
    pdb = os.path.join(prefix, "test_files", "input.pdb")
    zmat2 = os.path.join(prefix, "test_files", "zmat2.log")
    zmat3 = os.path.join(prefix, "test_files", "zmat3.log")

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
        if isinstance(ref, FileReader):
            ref = ref.atoms
        if isinstance(test, FileReader):
            test = test.atoms
        for r, t in zip(ref, test):
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
            if "link" not in skip and r.link_info != t.link_info:
                return False
            if "atomtype" not in skip and r.atomtype != t.atomtype:
                return False
            if "charge" not in skip and r.charge != t.charge:
                return False
            if "layer" not in skip and r.layer != t.layer:
                return False
            return True

    def test_read_xyz_structure(self):
        ref = self.xyz_matrix(TestFileIO.small_mol)
        mol = FileReader(TestFileIO.small_mol)
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

    def test_read_oniom_xyz(self):
        ref = self.xyz_matrix(self.oniom_xyz)
        mol = FileReader(self.oniom_xyz)
        for i, line in enumerate(ref):
            if i == 0:
                self.assertEqual(len(mol.atoms), int(line))
                continue
            if i == 1:
                self.assertEqual(mol.comment, line)
                continue
            n = i -2
            a = mol.atoms[n]
            self.assertEqual(a.element, line[0])
            dist = a.coords - np.array(line[1:4], dtype=float)
            dist = np.linalg.norm(dist)
            self.assertTrue(dist < 10 ** (-8))
            self.assertEqual(str(n+1), a.name)
            self.assertTrue(len(a.tags) ==0)
            self.assertEqual(a.layer, line[4])
            self.assertEqual(a.atomtype, line[5])
            self.assertEqual(a.charge, float(line[6]))
            if len(line) > 7:
                self.assertEqual(a.link_info["element"], line[7])
                self.assertEqual(a.link_info["atomtype"], line[8])
                self.assertEqual(float(a.link_info["charge"]), float(line[9]))
                self.assertEqual(int(a.link_info["connected"]), int(line[10]))

    def test_read_orca_out_structure(self):
        ref = FileReader(os.path.join(prefix, "ref_files", "orca_geom.xyz"))
        test = FileReader(os.path.join(prefix, "test_files", "orca_geom.out"))
        self.assertTrue(self.validate_atoms(ref, test, skip=["link", "atomtype", "charge", "layer"]))

    def test_read_orca_orbits(self):
        # this file can be tricky b/c MO coefficients are right next
        # to each other - no spaces between them
        test = FileReader(self.orca_orbit_file, just_geom=False)
        self.assertEqual(test.other["orbitals"].n_mos, 168)

    def test_read_psi4_dat_structure(self):
        ref = FileReader(os.path.join(prefix, "ref_files", "psi4_geom.xyz"))
        test = FileReader((self.psi4_output_file, "dat", None))
        self.assertTrue(self.validate_atoms(ref, test, skip=["link", "atomtype", "charge", "layer"]))

    def test_read_log_structure(self):
        ref = FileReader(
            os.path.join(prefix, "ref_files", "file_io_normal.xyz")
        )
        test = FileReader(os.path.join(prefix, "test_files", "normal.log"))
        self.assertTrue(self.validate_atoms(ref, test, skip=["link", "atomtype", "charge", "layer"]))

        ref = FileReader(
            os.path.join(prefix, "ref_files", "file_io_error.xyz")
        )
        test = FileReader(os.path.join(prefix, "test_files", "error.log"))
        self.assertTrue(self.validate_atoms(ref, test, skip=["link", "atomtype", "charge", "layer"]))

        ref = FileReader(os.path.join(prefix, "ref_files", "file_io_died.xyz"))
        test = FileReader(os.path.join(prefix, "test_files", "died.log"))
        self.assertTrue(self.validate_atoms(ref, test, skip=["link", "atomtype", "charge", "layer"]))
        
        ref = FileReader(os.path.join(prefix, "ref_files", "zmat_ref.xyz"))
        test = FileReader(os.path.join(prefix, "test_files", "zmat1.log"))
        self.assertTrue(self.validate_atoms(ref, test, skip=["link", "atomtype", "charge", "layer"]))
        
        # z-matrix things
        ref = FileReader(os.path.join(prefix, "ref_files", "zmat_ref.xyz"))
        input_ref = FileReader(os.path.join(prefix, "ref_files", "zmat_ref_init.xyz"))
        
        test = FileReader(os.path.join(prefix, "test_files", "zmat1.log"), get_all=True)
        self.assertTrue(self.validate_atoms(ref, test, skip=["link", "atomtype", "charge", "layer"]))        
        self.assertTrue(self.validate_atoms(input_ref, test.all_geom[0]["atoms"], skip=["link", "atomtype", "charge", "layer"]))
        
        test = FileReader(os.path.join(prefix, "test_files", "zmat2.log"), get_all=True)
        self.assertTrue(self.validate_atoms(ref, test, skip=["link", "atomtype", "charge", "layer"]))        
        self.assertTrue(self.validate_atoms(input_ref, test.all_geom[0]["atoms"], skip=["link", "atomtype", "charge", "layer"]))
        
        test = FileReader(os.path.join(prefix, "test_files", "zmat3.log"), get_all=True)
        self.assertTrue(self.validate_atoms(ref, test, skip=["link", "atomtype", "charge", "layer"]))        
        self.assertTrue(self.validate_atoms(input_ref, test.all_geom[0]["atoms"], skip=["link", "atomtype", "charge", "layer"]))

    def test_read_log_energies(self):
        """reading the correct energy from Gaussian output"""
        # mp2
        fr = FileReader(self.gaussian_mp2_file, just_geom=False)
        self.assertEqual(fr.other["energy"], -76.228479251471)
        
        # mp4
        fr = FileReader(self.gaussian_mp4_file, just_geom=False)
        self.assertEqual(fr.other["energy"], -76.240714654)
        
        # ccsd
        fr = FileReader(self.gaussian_ccsd_file, just_geom=False)
        self.assertEqual(fr.other["energy"], -76.238041859)
        
        # ccsd(t)
        fr = FileReader(self.gaussian_ccsd_t_file, just_geom=False)
        self.assertEqual(fr.other["energy"], -76.241082564)
        
        # double hybrid dft
        fr = FileReader(self.gaussian_dhdft_file, just_geom=False)
        self.assertEqual(fr.other["energy"], -76.353361915801)

    def test_read_com_info(self):
        """testing if we can read route info"""
        ref1 = {
            "method": "B3LYP/aug-cc-pVDZ",
            "temperature": 298.15,
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
            "temperature": 298.15,
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

    def test_read_pdb(self):
        """read rcsb pdb files"""
        ref = open(self.pdb, "r")
        test = Geometry(FileReader(self.pdb))
        line = ref.readline()
        while line.split()[0] != "ATOM":
            line = ref.readline()
        n = 0
        while line.strip() != "ENDMDL":
            if line.split()[0] in ("ATOM", "HETATM"):
                self.assertEqual(line[12:16].strip(), test.atoms[n].atomtype)
                self.assertEqual(line[76:78].strip(), test.atoms[n].element)
                self.assertTrue(float(line[30:38].strip()) == test.atoms[n].coords[0])
                self.assertTrue(float(line[38:46].strip()) == test.atoms[n].coords[1])
                self.assertTrue(float(line[46:54].strip()) == test.atoms[n].coords[2])
                self.assertEqual(line[17:20].strip(), test.atoms[n].res)
                n += 1
            line = ref.readline()

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

    def test_write_oniom_xyz(self):
        """write oniom xyz file"""
        #compares exact output
        geom = Geometry(self.oniom_com)

        ref = """108

Pd     0.25816   -1.83062   -0.54456 H  Pd   0.031338
O      0.88702    0.15529   -0.67153 H  o   -0.380776
O      2.95726    0.14240    0.18930 H  o   -0.345674
O      0.88263    6.96323   -1.02542 L  O   -0.228267
O      2.95678    7.01828   -0.45340 L  O   -0.220745
O     -1.42576   -1.20995    0.42466 H  o   -0.376012
O     -2.01071    0.56564   -0.86904 H  o   -0.304027
O     -5.62466    2.87256    4.54926 L  O   -0.226295
O     -5.97199    0.82390    5.15357 L  O   -0.223433
N      2.01402   -2.57176   -1.20940 H  n2  -0.228137
N     -0.62258   -1.70221   -2.48848 H  n2  -0.235079
N      1.92241    6.42430   -0.67752 L  na   0.081346
N     -5.44673    1.66133    4.43183 L  na   0.081350
C      2.39254   -2.56404   -2.50551 H  ca  -0.197199
H      1.81745   -2.18985   -3.16124 H  ha   0.322608
C      3.60595   -3.09398   -2.88984 H  ca  -0.272339
H      3.86703   -3.08569   -3.80390 H  ha   0.311492
C      4.43936   -3.64104   -1.92307 H  ca  -0.240717
H      5.26665   -4.03376   -2.17631 H  ha   0.314508
C      4.06735   -3.61381   -0.59559 H  ca  -0.356532
H      4.63835   -3.97697    0.07275 H  ha   0.312994
C      2.84163   -3.04849   -0.23808 H  ca   0.277773
C      2.34666   -2.86301    1.12425 H  ca   0.298952
C      3.03719   -3.23345    2.28358 H  ca  -0.448042
H      3.88138   -3.66150    2.21147 H  ha   0.299934
C      2.48655   -2.96803    3.51937 H  ca  -0.290065
H      2.95505   -3.21910    4.30663 H  ha   0.297104
C      1.25586   -2.33449    3.62263 H  ca  -0.285012
H      0.88734   -2.15639    4.47935 H  ha   0.306837
C      0.55438   -1.95990    2.47643 H  ca  -0.428953
H     -0.28951   -1.52861    2.54501 H  ha   0.349234
C      1.10964   -2.22559    1.24392 H  ca   0.153406
C     -0.59018   -0.64027   -3.31510 H  ca  -0.249483
H     -0.14692    0.15361   -3.03965 H  ha   0.371194
C     -1.19728   -0.68468   -4.56369 H  ca  -0.266227
H     -1.16410    0.06719   -5.14298 H  ha   0.314678
C     -1.85237   -1.84144   -4.95158 H  ca  -0.252075
H     -2.28115   -1.88578   -5.79976 H  ha   0.312094
C     -1.88209   -2.93300   -4.10273 H  ca  -0.379645
H     -2.32914   -3.73077   -4.35918 H  ha   0.307919
C     -1.24221   -2.84432   -2.85807 H  ca   0.188664
C     -1.20794   -3.91830   -1.85604 H  ca   0.255278
C     -1.81773   -5.16195   -2.04970 H  ca  -0.439192
H     -2.26668   -5.34421   -2.86608 H  ha   0.300371
C     -1.77010   -6.12670   -1.06012 H  ca  -0.298368
H     -2.19162   -6.96817   -1.19598 H  ha   0.301029
C     -1.10922   -5.86529    0.13103 H  ca  -0.312702
H     -1.07340   -6.53327    0.80645 H  ha   0.305421
C     -0.50237   -4.63843    0.33949 H  ca  -0.396017
H     -0.05768   -4.45993    1.16043 H  ha   0.338065
C     -0.54623   -3.67218   -0.65080 H  ca   0.203372
C      1.95354    0.68697   -0.28368 H  c2   0.237125
C      1.93638    2.20594   -0.39645 L  ca   0.241683 H  ha  0.241683 52
C      0.73507    2.88908   -0.63101 L  ca  -0.378438
H     -0.07215    2.40004   -0.73347 L  ha   0.375763
C      0.71707    4.27429   -0.71793 L  ca  -0.413382
H     -0.09344    4.74197   -0.87526 L  ha   0.348555
C      1.91422    4.95433   -0.56430 L  ca   0.294176
C      3.11638    4.30671   -0.33497 L  ca  -0.403284
H      3.92223    4.80042   -0.24029 L  ha   0.344050
C      3.11995    2.92577   -0.24287 L  ca  -0.405122
H      3.93433    2.46538   -0.07439 L  ha   0.347650
C     -2.05103   -0.09662    0.15779 H  c2   0.269523
C     -2.94356    0.34641    1.29958 L  ca   0.276535 H  ha  0.276535 63
C     -3.46736   -0.57286    2.21139 L  ca  -0.381969
H     -3.24857   -1.49246    2.13155 L  ha   0.346861
C     -4.30250   -0.14957    3.23109 L  ca  -0.408646
H     -4.67945   -0.77216    3.84132 L  ha   0.347815
C     -4.57690    1.20296    3.33655 L  ca   0.308276
C     -4.07041    2.14324    2.45466 L  ca  -0.402304
H     -4.27656    3.06613    2.55215 L  ha   0.350329
C     -3.25112    1.69965    1.42465 L  ca  -0.389842
H     -2.89678    2.32346    0.80127 L  ha   0.338697
C      6.04070    0.00786    0.26457 L  c3  -0.407317
H      4.95288    0.08222    0.30468 L  hc   0.278257
Cl     6.73701    1.55003    0.84431 L  Cl   0.043020
Cl     6.55148   -1.35307    1.31818 L  Cl   0.043021
Cl     6.50795   -0.29748   -1.43960 L  Cl   0.043021
C      8.23231   -6.21151   -0.70706 L  c3  -0.407317
H      9.30282   -6.37453   -0.62748 L  hc   0.278257
Cl     7.42299   -7.38079    0.37411 L  Cl   0.043020
Cl     7.90181   -4.52710   -0.20953 L  Cl   0.043021
Cl     7.77342   -6.48323   -2.41609 L  Cl   0.043021
C     -9.06870   -1.81313    2.40981 L  c3  -0.407317
H    -10.06953   -1.90054    2.82170 L  hc   0.278257
Cl    -8.06219   -0.92056    3.57112 L  Cl   0.043020
Cl    -8.45425   -3.48083    2.15255 L  Cl   0.043021
Cl    -9.20615   -0.94143    0.85017 L  Cl   0.043021
C      1.81361   10.17606   -1.14478 L  c3  -0.407317
H      1.71091    9.10081   -1.02966 L  hc   0.278257
Cl     2.95325   10.47447   -2.49374 L  Cl   0.043020
Cl     0.19589   10.84762   -1.51056 L  Cl   0.043021
Cl     2.45128   10.83513    0.39206 L  Cl   0.043021
C      3.55380    1.61359    4.83390 L  c3  -0.407317
H      4.32650    2.37589    4.80688 L  hc   0.278257
Cl     2.08695    2.36620    5.53370 L  Cl   0.043020
Cl     3.27083    1.04778    3.17204 L  Cl   0.043021
Cl     4.15817    0.29095    5.88679 L  Cl   0.043021
C     -3.46826    2.48597   -2.80050 L  c3  -0.407317
H     -3.06696    1.78365   -2.07099 L  hc   0.278257
Cl    -2.09322    3.08545   -3.78770 L  Cl   0.043020
Cl    -4.65478    1.61932   -3.82567 L  Cl   0.043021
Cl    -4.24519    3.82967   -1.91336 L  Cl   0.043021
C     -6.66958   -3.00503   -1.50620 L  c3  -0.407317
H     -7.21125   -2.78592   -0.59073 L  hc   0.278257
Cl    -6.13257   -1.45623   -2.21105 L  Cl   0.043020
Cl    -5.27109   -4.03749   -1.07483 L  Cl   0.043021
Cl    -7.79134   -3.86673   -2.60286 L  Cl   0.043021"""

        test = FileWriter.write_file(
            geom, style="xyz", outfile=False, oniom=all
        )

        for line1, line2 in zip(test.splitlines(), ref.splitlines()):
            self.assertEqual(line1.strip(), line2.strip())

    def test_write_inp(self):
        """write orca input file"""
        # like gaussian input files, this compares exact output

        geom = Geometry(self.small_mol)

        ref = """#comment line 1
#comment line 2
! PBE0 Opt Freq def2-SVP CPCM D3BJ
%freq
    Temp    298.15
end
%basis
    newGTO            C  "def2-TZVP" end
end
%cpcm
    smd    true
    SMDsolvent    "dichloromethane"
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
                [Basis("def2-SVP", ["!C"]), Basis("def2-TZVP", ["C"])]
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


        geom = Geometry(self.ts)

        ref = """#test_orbits.out
! M06L printMOs printBasis Opt def2-SVP
%geom
    Constraints
        {B  1  3 C}
        {B  2  5 C}
    end
end

*xyz 0 1
C     1.18854  -0.33581   0.91844
C     0.03710  -0.11108   1.65753
C     1.49009   0.55244  -0.12490
C    -1.46714  -0.56175   0.16760
C    -0.98763   0.39297  -0.71523
O     0.13435   0.25731  -1.31132
H     1.59783  -1.34820   0.87301
H    -0.30053  -0.83534   2.40220
H    -0.29409   0.91445   1.83715
H     2.32894   0.33997  -0.78844
H     1.29000   1.61694   0.01894
H    -1.12283  -1.59079   0.06594
H    -2.43357  -0.40885   0.65412
H    -1.46740   1.39172  -0.71711
*

"""

        theory = Theory(
            charge=0,
            multiplicity=1,
            method="M06-L",
            basis="def2-SVP",
            job_type=[OptimizationJob(constraints={"bonds":[["2", "4"], ["3", "6"]]})],
            comments=["test_orbits.out"],
        )

        kw_dict = {"simple": ["printMOs", "printBasis"]}

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
basis this_basis {
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

    def test_write_inq(self):
        """write q-chem input file"""
        # like gaussian input files, this compares exact output

        geom = Geometry(self.small_mol)

        ref = """$rem
    JOB_TYPE             =   OPT
    DFT_D                =   D3_BJ
    BASIS                =   General
    METHOD               =   PBE0
$end

$comments
    comment line 1
    comment line 2
$end

$basis
    H  0
    cc-pVDZ
    ****
    C  0
    aug-cc-pVDZ
    ****
    Cl 0
    aug-cc-pVDZ
    ****
    N  0
    aug-cc-pVDZ
    ****
    O  0
    aug-cc-pVDZ
    ****
$end

$molecule
    0 1
    C     -1.97696     -2.32718      0.00126
    C     -2.36814     -1.29554      0.85518
    C     -1.67136     -0.08735      0.85440
    C     -0.58210      0.08919      0.00026
    C     -0.19077     -0.94241     -0.85309
    C     -0.88848     -2.15056     -0.85289
    H     -3.22679     -1.43483      1.52790
    H     -1.98002      0.72606      1.52699
    H      0.66766     -0.80358     -1.52636
    H     -0.57992     -2.96360     -1.52585
    Cl     0.29699      1.61392     -0.00037
    N     -2.73689     -3.64357      0.00188
    O     -2.07823     -4.68230      0.00289
    O     -3.96579     -3.59263      0.00134
$end
"""

        theory = Theory(
            charge=0,
            multiplicity=1,
            method="PBE0",
            basis=BasisSet(
                [Basis("cc-pVDZ", ["H"]), Basis("aug-cc-pVDZ", ["!H"])]
            ),
            empirical_dispersion=EmpiricalDispersion("D3BJ"),
            job_type=[OptimizationJob()],
        )

        kw_dict = {ORCA_COMMENT: ["comment line 1", "comment line 2"]}

        test = FileWriter.write_inq(
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
                "avg_energy": 0.00027409943594479446,
                "temperature": 298.15,
                "energy": -79.18948,
                "energy_context": "E lowest                              :   -79.18948\n",
                "entropy": 0.010994255863855447,
                "free_energy": -0.003278038021735129,
                "best_pop": 70.368,
                "finished": True,
                "error": None,
            },
        )


if __name__ == "__main__":
    unittest.main()

