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

    def test_RMSD(self):
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

    def test_close_ring(self):
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
        

if __name__ == "__main__":
    unittest.main()
