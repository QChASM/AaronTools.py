#!/usr/bin/env python3
# testing for command line scripts
import os
import ssl
import sys
import unittest
from glob import glob
from subprocess import PIPE, Popen

import AaronTools
from AaronTools.atoms import Atom
from AaronTools.fileIO import FileReader
from AaronTools.geometry import CACTUS_HOST, OPSIN_HOST, Geometry
from AaronTools.test import TestWithTimer, prefix, rmsd_tol, validate
from AaronTools.test.test_geometry import is_close


class TestCLS(TestWithTimer):
    benz_NO2_Cl = os.path.join(prefix, "test_files", "benzene_1-NO2_4-Cl.xyz")

    benzene = os.path.join(prefix, "test_files", "benzene.xyz")

    pyridine = os.path.join(prefix, "test_files", "pyridine.xyz")

    chlorotoluene = os.path.join(prefix, "test_files", "chlorotoluene.xyz")
    chlorotoluene_ref = os.path.join(
        prefix, "ref_files", "chlorotoluene_180.xyz"
    )

    benzene_dimer = os.path.join(prefix, "test_files", "benzene_dimer.xyz")
    benzene_dimer_ref = os.path.join(
        prefix, "ref_files", "benzene_dimer_ref.xyz"
    )

    naphthalene = os.path.join(prefix, "ref_files", "naphthalene.xyz")

    tetrahydronaphthalene = os.path.join(
        prefix, "ref_files", "tetrahydronaphthalene.xyz"
    )

    pyrene = os.path.join(prefix, "ref_files", "pyrene.xyz")

    benz_OH_Cl = os.path.join(prefix, "test_files", "benzene_1-OH_4-Cl.xyz")

    frequencies = os.path.join(prefix, "test_files", "normal.log")

    rmsd_sort_1 = os.path.join(prefix, "test_files", "test_rmsd_sort1.xyz")
    rmsd_sort_2 = os.path.join(prefix, "test_files", "test_rmsd_sort2.xyz")

    g09_com_file = os.path.join(
        prefix, "test_files", "5a-sub1.R.ts1.Cf1.3.com"
    )
    g09_log_file = os.path.join(prefix, "test_files", "opt_normal.log")
    orca_out_file = os.path.join(prefix, "test_files", "orca_geom.out")
    psi4_dat_file = os.path.join(prefix, "test_files", "psi4-test.out")
    xyz_file = os.path.join(prefix, "test_files", "benzene.xyz")

    tm_simple = os.path.join(
        prefix, "test_files", "catalysts", "tm_single-lig.xyz"
    )

    t60 = os.path.join(prefix, "test_files", "torsion-60.xyz")
    t90 = os.path.join(prefix, "test_files", "torsion-90.xyz")

    opt_file_1 = os.path.join(prefix, "test_files", "opt_running.log")

    make_conf_1 = os.path.join(prefix, "test_files", "R-Quinox-tBu3.xyz")
    make_conf_ref_1 = sorted(
        glob(os.path.join(prefix, "ref_files", "make_conf_cls", "*.xyz"))
    )

    change_chir_1 = os.path.join(prefix, "test_files", "chiral_ring.xyz")
    change_chir_ref_1 = sorted(
        glob(
            os.path.join(prefix, "ref_files", "change_chirality_cls", "*.xyz")
        )
    )
    chiral_ring_mirror = os.path.join(
        prefix, "test_files", "chiral_ring_mirror.xyz"
    )
    cone_bidentate_2 = os.path.join(prefix, "test_files", "bpy.xyz")
    cone_bidentate_3 = os.path.join(prefix, "test_files", "dppe.xyz")

    aarontools_bin = os.path.join(os.path.dirname(AaronTools.__file__), "bin")
    # CLS stored in $PYTHONHOME/bin if installed via pip
    if aarontools_bin not in os.getenv("PATH"):
        proc = Popen(["pip", "show", "AaronTools"], stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()
        if out:
            tmp = os.path.split(aarontools_bin)
            while tmp[1] and tmp[1] != "lib":
                tmp = os.path.split(tmp[0])
            aarontools_bin = os.path.join(tmp[0], "bin")

    def test_environment(self):
        """is this AaronTools' bin in the path?"""
        path = os.getenv("PATH")
        self.assertTrue(self.aarontools_bin in path)

    # geometry measurement
    def test_angle(self):
        """measuring angles"""
        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "angle.py"),
            TestCLS.benz_NO2_Cl,
            "-m",
            "13",
            "12",
            "14",
        ]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        if len(err) != 0:
            raise RuntimeError(err)

        angle = float(out)
        self.assertTrue(is_close(angle, 124.752, 10 ** -2))

    def test_bond(self):
        """measuring bonds"""
        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "bond.py"),
            TestCLS.benzene,
            "-m",
            "1",
            "2",
        ]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        if len(err) != 0:
            raise RuntimeError(err)

        angle = float(out)
        self.assertTrue(is_close(angle, 1.3952, 10 ** -2))

    def test_dihedral(self):
        """measuring dihedrals"""
        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "dihedral.py"),
            TestCLS.benz_NO2_Cl,
            "-m",
            "13",
            "12",
            "1",
            "6",
        ]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        if len(err) != 0:
            raise RuntimeError(err)

        dihedral = float(out)
        self.assertTrue(is_close(dihedral, 45.023740, 10 ** -5))

    def test_rmsdAlign(self):
        """measuring rmsd"""
        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "rmsdAlign.py"),
            "-r",
            TestCLS.benz_NO2_Cl,
            TestCLS.benz_NO2_Cl,
            "--value",
        ]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        if len(err) != 0:
            raise RuntimeError(err)

        rmsd = float(out)
        self.assertTrue(is_close(rmsd, 0, 10 ** -5))

        # test sorting flag
        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "rmsdAlign.py"),
            "-r",
            TestCLS.rmsd_sort_1,
            TestCLS.rmsd_sort_2,
            "--value",
            "--sort",
        ]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        if len(err) != 0:
            raise RuntimeError(err)

        rmsd = float(out)
        self.assertTrue(rmsd < 0.1)

    def test_substitute(self):
        """test substitute.py"""
        ref = Geometry(TestCLS.benz_NO2_Cl)

        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "substitute.py"),
            TestCLS.benzene,
            "-s",
            "12=NO2",
            "-s",
            "11=Cl",
        ]
        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()
        # print(out.decode("utf-8"))
        # print(err.decode("utf-8"))
        fr = FileReader(("out", "xyz", out.decode("utf-8")))
        mol = Geometry(fr)
        self.assertTrue(validate(mol, ref))

        # I don't want to run these tests anymore
        # sometimes the server timeout stuff doesn't seem to work and
        # the test hangs for > 30 seconds
        # - Tony
        # # don't run smiles/iupac tests if we can't connect to the host site
        # try:
        #     for host in [CACTUS_HOST, OPSIN_HOST]:
        #         response = urllib.request.urlopen(
        #             host, context=ssl.SSLContext()
        #         )
        #         # name resolved, but something wrong with server
        #         if response.status >= 5:
        #             return
        # except urllib.error.URLError:
        #     # can't resolve name or some other connectivity error
        #     return
        #
        # args = [
        #     sys.executable,
        #     os.path.join(self.aarontools_bin, "substitute.py"),
        #     TestCLS.benzene,
        #     "-s",
        #     "12=smiles:O=[N.]=O",
        #     "-s",
        #     "11=iupac:chloro",
        # ]
        # proc = Popen(args, stdout=PIPE, stderr=PIPE)
        # out, err = proc.communicate()
        # fr = FileReader(("out", "xyz", out.decode("utf-8")))
        # mol = Geometry(fr)
        # self.assertTrue(validate(mol, ref, thresh=2e-1))

    def test_closeRing(self):
        """test closeRing.py"""
        ref1 = Geometry(TestCLS.naphthalene)

        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "closeRing.py"),
            TestCLS.benzene,
            "-r",
            "7",
            "8",
            "benzene",
        ]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        if len(err) != 0:
            raise RuntimeError(err)

        fr = FileReader(("out", "xyz", out.decode("utf-8")))
        mol = Geometry(fr)
        rmsd = mol.RMSD(ref1, sort=True)
        self.assertTrue(rmsd < rmsd_tol(ref1, superLoose=True))

    def test_mirror(self):
        """test mirror.py"""
        ref1 = Geometry(TestCLS.chiral_ring_mirror)

        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "mirror.py"),
            TestCLS.change_chir_1,
            "-xy",
        ]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        if len(err) != 0:
            raise RuntimeError(err)

        fr = FileReader(("out", "xyz", out.decode("utf-8")))
        mol = Geometry(fr)
        rmsd = mol.RMSD(ref1, sort=True)
        self.assertTrue(rmsd < rmsd_tol(ref1, superLoose=True))

    def test_grabThermo(self):
        """test grabThermo.py"""
        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "grabThermo.py"),
            TestCLS.frequencies,
        ]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        if len(err) != 0:
            raise RuntimeError(err)

        ref = """electronic energy of test_files/normal.log = -1856.018658 Eh
    E+ZPE             = -1855.474686 Eh  (ZPE = 0.543972)
thermochemistry from test_files/normal.log at 298.00 K:
    H(RRHO)           = -1855.440616 Eh  (dH = 0.578042)
    G(RRHO)           = -1855.538017 Eh  (dG = 0.480642)
  quasi treatments for entropy (w0=100.0 cm^-1):
    G(Quasi-RRHO)     = -1855.532805 Eh  (dG = 0.485854)
    G(Quasi-harmonic) = -1855.532510 Eh  (dG = 0.486148)
"""
        # strip b/c windows adds \r to the end of lines
        out_list = out.decode("utf-8").splitlines()
        ref_list = ref.splitlines()

        # can't test all the lines b/c paths might be different
        # test sp energy
        self.assertTrue(out_list[0][-16:] == ref_list[0][-16:])
        # test thermochem
        for i in [1, 3, 4, 6, 7]:
            self.assertTrue(out_list[i][-34:] == ref_list[i][-34:])

        # test regular output with sp
        # sp is the same as the thermo file
        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "grabThermo.py"),
            TestCLS.frequencies,
            "-sp",
            TestCLS.frequencies,
        ]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        if len(err) != 0:
            raise RuntimeError(err)

        out_list = out.decode("utf-8").splitlines()
        self.assertTrue(out_list[0][-16:] == ref_list[0][-16:])
        for i in [1, 3, 4, 6, 7]:
            self.assertTrue(out_list[i][-34:] == ref_list[i][-34:])

        # test CSV w/o sp file
        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "grabThermo.py"),
            TestCLS.frequencies,
            "-csv",
        ]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        if len(err) != 0:
            raise RuntimeError(err)

        ref_csv = """E,E+ZPE,H(RRHO),G(RRHO),G(Quasi-RRHO),G(Quasi-harmonic),ZPE,dH(RRHO),dG(RRHO),dG(Quasi-RRHO),dG(Quasi-harmonic),SP_File,Thermo_File
-1856.018658,-1855.474686,-1855.440616,-1855.538017,-1855.532805,-1855.532510,0.543972,0.578042,0.480642,0.485854,0.486148,test_files/normal.log,test_files/normal.log"""

        out_list = out.decode("utf-8").splitlines()
        ref_list = ref_csv.splitlines()

        out_list = out.decode("utf-8").splitlines()
        ref_list = ref_csv.splitlines()

        # test CSV with sp file
        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "grabThermo.py"),
            TestCLS.frequencies,
            "-csv",
            "-sp",
            TestCLS.frequencies,
        ]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        if len(err) != 0:
            raise RuntimeError(err)

        out_list = out.decode("utf-8").splitlines()
        ref_list = ref_csv.splitlines()

        self.assertTrue(out_list[0] == ref_list[0])
        self.assertTrue(
            out_list[1].split(",")[:-2] == ref_list[1].split(",")[:-2]
        )

        # test CSV with looking in subdirectories
        filename = os.path.basename(TestCLS.frequencies)
        directory = os.path.join(prefix, "test_files")
        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "grabThermo.py"),
            directory,
            "-r",
            filename,
            "-csv",
            "-sp",
            filename,
        ]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        if len(err) != 0:
            raise RuntimeError(err)

        out_list = out.decode("utf-8").splitlines()
        ref_list = ref_csv.splitlines()

        self.assertTrue(out_list[0] == ref_list[0])
        self.assertTrue(
            out_list[1].split(",")[:-2] == ref_list[1].split(",")[:-2]
        )

    def test_printXYZ(self):
        """test printXYZ.py"""
        # for each test, the rmsd tolerance is determined based on the number of atoms and
        # the precision we use when printing xyz files
        # test xyz file
        ref_xyz = Geometry(TestCLS.xyz_file)

        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "printXYZ.py"),
            TestCLS.xyz_file,
        ]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        if len(err) != 0:
            raise RuntimeError(err)

        fr = FileReader(("out", "xyz", out.decode("utf-8")))
        mol = Geometry(fr)
        rmsd = mol.RMSD(ref_xyz, align=True)
        self.assertTrue(rmsd < len(ref_xyz.atoms) * (3 * 1e-5))

        # test gaussian input file
        ref_com = Geometry(TestCLS.g09_com_file)

        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "printXYZ.py"),
            TestCLS.g09_com_file,
        ]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        if len(err) != 0:
            raise RuntimeError(err)

        fr = FileReader(("out", "xyz", out.decode("utf-8")))
        mol = Geometry(fr)
        rmsd = mol.RMSD(ref_com, align=True)
        self.assertTrue(rmsd < len(ref_com.atoms) * (3 * 1e-5))

        # test gaussian output file
        ref_log = Geometry(TestCLS.g09_log_file)

        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "printXYZ.py"),
            TestCLS.g09_log_file,
        ]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        if len(err) != 0:
            raise RuntimeError(err)

        fr = FileReader(("out", "xyz", out.decode("utf-8")))
        mol = Geometry(fr)
        rmsd = mol.RMSD(ref_log, align=True)
        self.assertTrue(rmsd < len(ref_log.atoms) * (3 * 1e-5))

        # test orca output file
        ref_out = Geometry(TestCLS.orca_out_file)

        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "printXYZ.py"),
            TestCLS.orca_out_file,
        ]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        if len(err) != 0:
            raise RuntimeError(err)

        fr = FileReader(("out", "xyz", out.decode("utf-8")))
        mol = Geometry(fr)
        rmsd = mol.RMSD(ref_out, align=True)
        self.assertTrue(rmsd < len(ref_out.atoms) * (3 * 1e-5))

        # test psi4 output files and format flat
        ref_dat = Geometry(FileReader((TestCLS.psi4_dat_file, "dat", None)))

        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "printXYZ.py"),
            TestCLS.psi4_dat_file,
            "-if",
            "dat",
        ]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        if len(err) != 0:
            raise RuntimeError(err)

        fr = FileReader(("out", "xyz", out.decode("utf-8")))
        mol = Geometry(fr)
        rmsd = mol.RMSD(ref_dat, align=True)
        self.assertTrue(rmsd < len(ref_dat.atoms) * (3 * 1e-5))

    def test_changeElement(self):
        """test changeElement.py"""
        ref = Geometry(TestCLS.pyridine)

        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "changeElement.py"),
            TestCLS.benzene,
            "-e",
            "1=N",
            "-c",
        ]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        if len(err) != 0:
            print(out.decode("utf-8"))
            print(err.decode("utf-8"))
            raise RuntimeError(err.decode("utf-8"))

        fr = FileReader(("out", "xyz", out.decode("utf-8")))
        mol = Geometry(fr)
        self.assertTrue(validate(ref, mol, thresh="loose"))

    def test_rotate(self):
        """test rotate.py"""
        ref = Geometry(TestCLS.chlorotoluene_ref)

        # range of targets
        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "rotate.py"),
            TestCLS.chlorotoluene,
            "-b",
            "3",
            "12",
            "-c",
            "3",
            "-t",
            "12-15",
            "-a",
            "180",
        ]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        if len(err) != 0:
            raise RuntimeError(err)

        fr = FileReader(("out", "xyz", out.decode("utf-8")))
        mol = Geometry(fr)
        rmsd = mol.RMSD(ref, align=True)
        self.assertTrue(rmsd < rmsd_tol(ref))

        # enumerate all targets
        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "rotate.py"),
            TestCLS.chlorotoluene,
            "-b",
            "3",
            "12",
            "-c",
            "3",
            "-t",
            "12,13,14,15",
            "-a",
            "180",
        ]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        if len(err) != 0:
            raise RuntimeError(err)

        fr = FileReader(("out", "xyz", out.decode("utf-8")))
        mol = Geometry(fr)
        rmsd = mol.RMSD(ref, align=True)
        self.assertTrue(rmsd < rmsd_tol(ref))

        # rotate all atom by 180 - rmsd should be basically 0
        ref2 = Geometry(TestCLS.chlorotoluene)

        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "rotate.py"),
            TestCLS.chlorotoluene,
            "-x",
            "x",
            "-a",
            "180",
        ]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        if len(err) != 0:
            raise RuntimeError(err)

        fr = FileReader(("out", "xyz", out.decode("utf-8")))
        mol = Geometry(fr)
        rmsd = mol.RMSD(ref2, align=True)
        self.assertTrue(rmsd < rmsd_tol(ref2))

        # rotate one fragment
        ref3 = Geometry(TestCLS.benzene_dimer_ref)

        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "rotate.py"),
            TestCLS.benzene_dimer,
            "-p",
            "1-12",
            "-c",
            "1-12",
            "-f",
            "1",
            "-a",
            "10",
        ]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        if len(err) != 0:
            raise RuntimeError(err)

        fr = FileReader(("out", "xyz", out.decode("utf-8")))
        mol = Geometry(fr)
        rmsd = mol.RMSD(ref3, align=True)
        self.assertTrue(rmsd < rmsd_tol(ref3))

    def test_mapLigand(self):
        """test mapLigand.py"""
        ref = Geometry(os.path.join(prefix, "ref_files", "lig_map_3.xyz"))

        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "mapLigand.py"),
            TestCLS.tm_simple,
            "-l",
            "35,36=S-tBu-BOX",
        ]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        if len(err) != 0:
            raise RuntimeError(err)

        fr = FileReader(("out", "xyz", out.decode("utf-8")))
        mol = Geometry(fr)
        self.assertTrue(validate(mol, ref, thresh="loose", debug=False))

    def test_interpolate(self):
        """test interpolate.py
        assumes current working directory is writable b/c interpolate doesn't
        print structures to stdout"""
        ref = Geometry(
            os.path.join(prefix, "ref_files", "torsion_interpolation.xyz")
        )

        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "interpolate.py"),
            TestCLS.t60,
            TestCLS.t90,
            "-t",
            "0.40",
            "-u",
        ]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        if len(err) != 0:
            raise RuntimeError(err)

        mol = Geometry("traj-0.xyz")
        os.remove("traj-0.xyz")
        rmsd = mol.RMSD(ref, align=True, sort=True)
        self.assertTrue(rmsd < rmsd_tol(ref, superLoose=True))

    def test_makeConf(self):
        """test makeConf.py"""

        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "makeConf.py"),
            TestCLS.make_conf_1,
        ]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        if len(err) != 0:
            raise RuntimeError(err)

        fr = FileReader(("out", "xyz", out.decode("utf-8")), get_all=True)

        for step, ref in zip(fr.all_geom, self.make_conf_ref_1):
            geom = None
            for item in step:
                if isinstance(item, list) and all(
                    isinstance(a, Atom) for a in item
                ):
                    geom = Geometry(item)

            if geom is None:
                raise RuntimeError("an output is missing atoms")

            ref_geom = Geometry(ref)
            rmsd = ref_geom.RMSD(geom)
            self.assertTrue(rmsd < rmsd_tol(ref_geom))

    def test_changeChirality(self):
        """test changeChirality.py"""

        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "changeChirality.py"),
            TestCLS.change_chir_1,
            "--diastereomers",
        ]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        if len(err) != 0:
            raise RuntimeError(err)

        fr = FileReader(("out", "xyz", out.decode("utf-8")), get_all=True)

        for step, ref in zip(fr.all_geom, self.change_chir_ref_1):
            geom = None
            for item in step:
                if isinstance(item, list) and all(
                    isinstance(a, Atom) for a in item
                ):
                    geom = Geometry(item)

            if geom is None:
                raise RuntimeError("an output is missing atoms")

            ref_geom = Geometry(ref)
            rmsd = ref_geom.RMSD(geom)
            self.assertTrue(rmsd < rmsd_tol(ref_geom))

    def test_grabStatus(self):
        """test grabStatus.py"""

        # gaussian file that is not finished
        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "grabStatus.py"),
            TestCLS.opt_file_1,
        ]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        if len(err) != 0:
            raise RuntimeError(err)

        # ref = """                      Filename        Max Disp       Max Force        RMS Disp       RMS Force
        #     test_files/opt_running.log     1.66e+00/NO     3.61e+00/NO     2.79e-01/NO     2.55e-01/NO  not finished"""

        ref = """                      Filename    Step        Max Disp       Max Force        RMS Disp       RMS Force
            test_files/opt_running.log       4     1.66e+00/NO     3.61e+00/NO     2.79e-01/NO     2.55e-01/NO  not finished"""

        ref_lines = ref.splitlines()
        ref_status_line = ref_lines[1]

        lines = out.decode("utf-8").splitlines()
        test_line = lines[1]

        # don't include filename in test b/c that will be different
        for ref_item, test_item in zip(
            ref_status_line.split()[-7:], test_line.split()[-7:]
        ):
            # print(ref_item, test_item)
            self.assertTrue(ref_item == test_item)

    def test_substituentSterimol(self):
        """test substituentSterimol.py"""

        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "substituentSterimol.py"),
            TestCLS.benzene,
            "-s",
            "1",
            "-a" "12",
        ]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        if len(err) != 0:
            raise RuntimeError(err)

        ref = """B1      B2      B3      B4      B5      L       file
        1.70    1.70    3.36    3.36    3.36    6.78    test_files/benzene.xyz

        """

        ref_lines = ref.splitlines()
        ref_status_line = ref_lines[1]

        lines = out.decode("utf-8").splitlines()
        test_line = lines[1]

        # don't include filename in test b/c that will be different
        for ref_item, test_item in zip(
            ref_status_line.split()[:3], test_line.split()[:3]
        ):
            if ref_item != test_item:
                print(ref_item, test_item)
            self.assertTrue(ref_item == test_item)

    def test_coneAngle(self):
        """test coneAngle.py"""

        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "coneAngle.py"),
            "-r",
            "bondi",
            TestCLS.cone_bidentate_3,
            "-k",
            "2,3",
            "-m",
            "exact",
        ]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        angle = float(out)
        self.assertTrue(abs(angle - 218.6) <= 0.1)

        args = [
            sys.executable,
            os.path.join(self.aarontools_bin, "coneAngle.py"),
            "-r",
            "bondi",
            TestCLS.cone_bidentate_2,
            "-k",
            "2,3",
            "-m",
            "exact",
        ]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        angle = float(out)
        self.assertTrue(abs(angle - 194.6) <= 0.1)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestCLS("test_grabThermo"))
    return suite


ONLYSOME = False

if __name__ == "__main__" and ONLYSOME:
    runner = unittest.TextTestRunner()
    runner.run(suite())
elif __name__ == "__main__":
    unittest.main()
