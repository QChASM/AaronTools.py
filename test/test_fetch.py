#! /usr/bin/env python3
"""Testing for getting geometries, substituents, and rings from
an IUPAC name or a SMILES string"""
import os
import unittest

import numpy as np
from AaronTools.geometry import Geometry
from AaronTools.ring import Ring
from AaronTools.substituent import Substituent
from AaronTools.test import TestWithTimer, prefix, rmsd_tol, validate


class TestFromString(TestWithTimer):
    """
    these only get run if RDKit is installed
    molecules can be fetched from CATCVS, but there can be
    discrepencies that make these tests "fail"
    """

    COCH3 = Substituent("COCH3")
    NO2 = Substituent("NO2")
    benzene = Ring("benzene")
    chiral_geom = Geometry(
        os.path.join(prefix, "test_files", "chiral_ring.xyz")
    )

    def is_COCH3(self, sub):
        ref = TestFromString.COCH3
        self.assertTrue(validate(sub, ref, thresh=5e-2, heavy_only=True))

    def is_NO2(self, sub):
        ref = TestFromString.NO2
        self.assertTrue(validate(sub, ref, thresh=2e-1))

    def test_substituent(self):
        try:
            import rdkit

            sub = Substituent.from_string("acetyl", form="iupac")
            self.is_COCH3(sub)

            sub = Substituent.from_string("nitro", form="iupac")
            self.is_NO2(sub)

            sub = Substituent.from_string("O=[N.]=O", form="smiles")
            self.is_NO2(sub)

            sub = Substituent.from_string("O=[N]=O", form="smiles")
            self.is_NO2(sub)

        except ImportError:
            pass

    def test_geometry(self):
        try:
            import rdkit

            geom = Geometry.from_string(
                "(1R,2R)-1-Chloro-2-methylcyclohexane", form="iupac"
            )
            ref = TestFromString.chiral_geom
            # really loose threshhold b/c rdkit can give a boat cyclohexane...
            self.assertTrue(
                validate(geom, ref, thresh=0.35, heavy_only=True, debug=False)
            )

        except:
            pass

    def test_ring(self):
        try:
            import rdkit

            ring = Ring.from_string(
                "benzene", end_length=1, end_atom="C", form="iupac"
            )
            ref = self.benzene
            self.assertTrue(validate(ring, ref, thresh="loose"))

        except ImportError:
            pass


if __name__ == "__main__":
    unittest.main()
