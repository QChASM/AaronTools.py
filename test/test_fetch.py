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

    def is_COCH3(self, sub, thresh=0.03):
        ref = TestFromString.COCH3
        ref.refresh_connected()
        sub.refresh_connected()
        ref.refresh_ranks()
        ref.refresh_ranks()
        ref.atoms = ref.reorder(start=ref.atoms[0])[0]
        sub.atoms = sub.reorder(start=sub.atoms[0])[0]
        self.assertTrue(
            validate(
                sub,
                ref,
                thresh=thresh,
                heavy_only=True,
                sort=False,
                debug=False,
            )
        )

    def is_NO2(self, sub):
        ref = TestFromString.NO2
        self.assertTrue(validate(sub, ref, thresh=2e-1))

    def test_substituent(self):
        try:
            import rdkit

            sub = Substituent.from_string(
                "acetyl", form="iupac", strict_use_rdkit=True
            )
            self.is_COCH3(sub)

            with self.assertLogs(Substituent.LOG, level="WARNING"):
                sub = Substituent.from_string(
                    "nitro", form="iupac", strict_use_rdkit=True
                )
                self.is_NO2(sub)

            sub = Substituent.from_string(
                "O=[N.]=O", form="smiles", strict_use_rdkit=True
            )
            self.is_NO2(sub)

            sub = Substituent.from_string(
                "O=[N]=O", form="smiles", strict_use_rdkit=True
            )
            self.is_NO2(sub)

        except (ImportError, ModuleNotFoundError):
            # I still want to test CACTVS things because sometimes they change stuff
            # that breaks our stuff
            if any(
                user == os.getenv("USER", os.getenv("USERNAME", False))
                for user in ["ajs99778", "normn"]
            ):
                sub = Substituent.from_string("acetyl", form="iupac")
                print(sub.write(outfile=False))
                self.is_COCH3(sub, thresh=0.3)

                sub = Substituent.from_string("nitro", form="iupac")
                print(sub.write(outfile=False))
                self.is_NO2(sub)

                sub = Substituent.from_string("O=[N.]=O", form="smiles")
                print(sub.write(outfile=False))
                self.is_NO2(sub)

                sub = Substituent.from_string("O=[N]=O", form="smiles")
                print(sub.write(outfile=False))
                self.is_NO2(sub)

            else:
                self.skipTest("RDKit not installed, CACTVS is not tested")

    def test_geometry(self):
        try:
            import rdkit

            geom = Geometry.from_string(
                "(1R,2R)-1-Chloro-2-methylcyclohexane", form="iupac"
            )
            ref = TestFromString.chiral_geom
            # really loose threshhold b/c rdkit can give a boat cyclohexane...
            self.assertTrue(validate(geom, ref, thresh=0.35, heavy_only=True))

        except (ImportError, ModuleNotFoundError):
            if any(
                user == os.getenv("USER", os.getenv("USERNAME", False))
                for user in ["ajs99778", "normn"]
            ):
                geom = Geometry.from_string(
                    "(1R,2R)-1-Chloro-2-methylcyclohexane", form="iupac"
                )
                ref = TestFromString.chiral_geom
                # really loose threshhold b/c rdkit can give a boat cyclohexane...
                self.assertTrue(
                    validate(geom, ref, thresh=0.35, heavy_only=True)
                )
            else:
                self.skipTest("RDKit not installed, CACTVS is not tested")

    def test_ring(self):
        try:
            import rdkit

            ring = Ring.from_string(
                "benzene", end_length=1, end_atom="C", form="iupac"
            )
            ref = self.benzene
            self.assertTrue(validate(ring, ref, thresh="loose"))

        except ImportError:
            if any(
                user == os.getenv("USER", os.getenv("USERNAME", False))
                for user in ["ajs99778", "normn"]
            ):
                ring = Ring.from_string(
                    "benzene", end_length=1, end_atom="C", form="iupac"
                )
                print(ring.comment)
                ref = self.benzene
                self.assertTrue(
                    validate(ring, ref, thresh="loose", debug=True)
                )

            else:
                self.skipTest("RDKit not installed, CACTVS is not tested")


if __name__ == "__main__":
    unittest.main()
