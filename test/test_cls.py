#!/usr/bin/env python3
# testing for command line scripts
import os
import unittest
import numpy as np

from copy import copy

from subprocess import Popen, PIPE

from io import StringIO

import AaronTools
from AaronTools.atoms import Atom
from AaronTools.fileIO import FileReader, FileWriter
from AaronTools.geometry import Geometry
from AaronTools.ring import Ring
from AaronTools.substituent import Substituent
from AaronTools.test import TestWithTimer, prefix, rmsd_tol
from AaronTools.test.test_geometry import is_close


class TestCLS(TestWithTimer):
    benz_NO2_Cl = os.path.join(prefix, "test_files/benzene_1-NO2_4-Cl.xyz")
    benzene = os.path.join(prefix, "test_files", "benzene.xyz")
    pentane = os.path.join(prefix, "test_files", "pentane.xyz")
    naphthalene = os.path.join(prefix, "ref_files", "naphthalene.xyz")
    tetrahydronaphthalene = os.path.join(
        prefix, "ref_files", "tetrahydronaphthalene.xyz"
    )
    pyrene = os.path.join(prefix, "ref_files", "pyrene.xyz")
    benz_OH_Cl = os.path.join(prefix, "test_files", "benzene_1-OH_4-Cl.xyz")
    frequencies = os.path.join(prefix, "test_files", "normal.log")

    aarontools_bin = os.path.join(os.path.dirname(AaronTools.__file__), "bin")
   
    def test_environment(self):
        """is this AaronTools' bin in the path?"""
        path = os.getenv('PATH')

        self.assertTrue(self.aarontools_bin in path)
    
    # geometry measurement
    def test_angle(self):
        """measuring angles"""
        args = [os.path.join(self.aarontools_bin, "angle.py"), \
                TestCLS.benz_NO2_Cl, \
                "-m", "13", "12", "14"]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()
        
        self.assertTrue(len(err) == 0)
        
        angle = float(out)
        self.assertTrue(is_close(angle, 124.752, 10 ** -2))

    def test_dihedral(self):
        """measuring dihedrals"""
        args = [os.path.join(self.aarontools_bin, "dihedral.py"), \
                TestCLS.benz_NO2_Cl, \
                "-m", "13", "12", "1", "6"]
        
        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()
        
        self.assertTrue(len(err) == 0)
        
        dihedral = float(out)
        self.assertTrue(is_close(dihedral, 45.023740, 10 ** -5))

    def test_rmsdAlign(self):
        """measuring rmsd"""
        args = [os.path.join(self.aarontools_bin, "rmsdAlign.py"), \
                "-r", TestCLS.benz_NO2_Cl, \
                TestCLS.benz_NO2_Cl, \
                "--value"]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()
        
        self.assertTrue(len(err) == 0)
        
        rmsd = float(out)
        self.assertTrue(is_close(rmsd, 0, 10 ** -5))

    def test_substitute(self):
        ref = Geometry(TestCLS.benz_NO2_Cl)

        args = [os.path.join(self.aarontools_bin, "substitute.py"), \
                TestCLS.benzene, \
                "-s", "12=NO2", "11=Cl"]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()
       
        self.assertTrue(len(err) == 0)

        fr = FileReader(("out", "xyz", out.decode('utf-8')))
        mol = Geometry(fr)
        rmsd = mol.RMSD(ref, align=True)
        self.assertTrue(rmsd < rmsd_tol(ref))

    def test_closeRing(self):
        ref1 = Geometry(TestCLS.naphthalene)

        args = [os.path.join(self.aarontools_bin, "closeRing.py"), \
                TestCLS.benzene, \
                "-r", "7", "8", "benzene"]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()
       
        self.assertTrue(len(err) == 0)

        fr = FileReader(("out", "xyz", out.decode('utf-8')))
        mol = Geometry(fr)
        rmsd = mol.RMSD(ref1, align=True)
        self.assertTrue(rmsd < rmsd_tol(ref1, superLoose=True))

    def test_grabThermo(self):
        args = [os.path.join(self.aarontools_bin, "grabThermo.py"), \
                TestCLS.frequencies]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()
       
        self.assertTrue(len(err) == 0)
        
        ref = """electronic energy of test_files/normal.log = -1856.018658 Eh
    ZPE               = -1855.474686 Eh  (dZPE = 0.543972)
thermochemistry from test_files/normal.log at 298.00 K:
    H(RRHO)           = -1855.440616 Eh  (dH = 0.578042)
    G(RRHO)           = -1855.538017 Eh  (dG = 0.480642)
  quasi treatments for entropy (w0=100.0 cm^-1):
    G(Quasi-RRHO)     = -1855.532805 Eh  (dG = 0.485854)
    G(Quasi-harmonic) = -1855.532510 Eh  (dG = 0.486148)
"""
        out_list = out.decode('utf-8').split('\n')
        ref_list = ref.split('\n')

        #can't test all the lines b/c paths might be different
        #test sp energy
        self.assertTrue(out_list[0][-16:] == ref_list[0][-16:])
        #test thermochem
        for i in [1, 3, 4, 6, 7]:
            self.assertTrue(out_list[i][-34:] == ref_list[i][-34:])
       

        #test regular output with sp
        #sp is the same as the thermo file
        args = [os.path.join(self.aarontools_bin, "grabThermo.py"), \
                TestCLS.frequencies, '-sp', TestCLS.frequencies]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()
      
        self.assertTrue(len(err) == 0)
        
        self.assertTrue(out_list[0][-16:] == ref_list[0][-16:])
        for i in [1, 3, 4, 6, 7]:
            self.assertTrue(out_list[i][-34:] == ref_list[i][-34:])
       

        #test CSV w/o sp file
        args = [os.path.join(self.aarontools_bin, "grabThermo.py"), \
                TestCLS.frequencies, '-csv']

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()
       
        self.assertTrue(len(err) == 0)

        ref_csv = """E,ZPE,H(RRHO),G(RRHO),G(Quasi-RRHO),G(Quasi-harmonic),dZPE,dH(RRHO),dG(RRHO),dG(Quasi-RRHO),dG(Quasi-harmonic),SP_File,Thermo_File
-1856.018658,-1855.474686,-1855.440616,-1855.538017,-1855.532805,-1855.532510,0.543972,0.578042,0.480642,0.485854,0.486148,test_files/normal.log,test_files/normal.log"""
        
        out_list = out.decode('utf-8').split('\n')
        ref_list = ref_csv.split('\n')

        self.assertTrue(out_list[0] == ref_list[0])
        self.assertTrue(out_list[1].split(',')[:-2] == ref_list[1].split(',')[:-2])

        
        #test CSV with sp file
        args = [os.path.join(self.aarontools_bin, "grabThermo.py"), \
                TestCLS.frequencies, '-csv', '-sp', TestCLS.frequencies]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()
       
        self.assertTrue(len(err) == 0)

        out_list = out.decode('utf-8').split('\n')
        ref_list = ref_csv.split('\n')

        self.assertTrue(out_list[0] == ref_list[0])
        self.assertTrue(out_list[1].split(',')[:-2] == ref_list[1].split(',')[:-2])

        
        #test CSV with looking in subdirectories
        filename = os.path.basename(TestCLS.frequencies)
        directory = os.path.join(prefix, "test_files")
        args = [os.path.join(self.aarontools_bin, "grabThermo.py"), \
                directory, '-r', filename, '-csv', '-sp', filename]

        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()
       
        self.assertTrue(len(err) == 0)

        out_list = out.decode('utf-8').split('\n')
        ref_list = ref_csv.split('\n')

        self.assertTrue(out_list[0] == ref_list[0])
        self.assertTrue(out_list[1].split(',')[:-2] == ref_list[1].split(',')[:-2])


if __name__ == "__main__":
    unittest.main()
