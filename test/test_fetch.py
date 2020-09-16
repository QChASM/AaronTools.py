#! /usr/bin/env python3
"""Testing for getting geometries, substituents, and rings from
an IUPAC name or a SMILES string"""
import os
import unittest

import numpy as np

from AaronTools.geometry import Geometry
from AaronTools.substituent import Substituent
from AaronTools.ring import Ring
from AaronTools.test import TestWithTimer, prefix, rmsd_tol, validate


class TestFromString(TestWithTimer):
    COCH3 = Substituent("COCH3")
    NO2 = Substituent("NO2")
    benzene = Ring("benzene")
    chiral_geom = Geometry(
        os.path.join(prefix, "test_files/chiral_ring.xyz")
    )

    def is_COCH3(self, sub):
        ref = TestFromString.COCH3
        self.assertTrue(validate(sub, ref, thresh=5e-2, heavy_only=True))

    def is_NO2(self, sub):
        ref = TestFromString.NO2
        self.assertTrue(validate(sub, ref, thresh=2e-1))

    def test_substituent(self):
        sub = Substituent.from_string("acetyl", form='iupac')
        self.is_COCH3(sub)

        sub = Substituent.from_string("nitro", form='iupac')
        self.is_NO2(sub)

        sub = Substituent.from_string("O=[N.]=O", form='smiles')
        self.is_NO2(sub)

        sub = Substituent.from_string("O=[N]=O", form='smiles')
        self.is_NO2(sub)

    def test_geometry(self):
        geom = Geometry.from_string("(1R,2R)-1-Chloro-2-methylcyclohexane", form="iupac")
        ref = TestFromString.chiral_geom
        self.assertTrue(validate(geom, ref, thresh="loose", heavy_only=True))

    def test_ring(self):
        ring = Ring.from_string("benzene", end_length=1, end_atom='C', form="iupac")
        ref = self.benzene
        self.assertTrue(validate(ring, ref, thresh="loose"))


if __name__ == "__main__":
    unittest.main()
