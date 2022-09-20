#!/usr/bin/env python3
import json
import os
import unittest

import numpy as np
from AaronTools import addlogger
from AaronTools.config import Config
from AaronTools.test import TestWithTimer, prefix
from AaronTools.theory import Theory, ImplicitSolvent, OptimizationJob


@addlogger
class TestConfig(TestWithTimer):
    LOG = None
    USER_SPECIFIC = ["metadata", "infile", (".*", ".*_dir")]
    config_list = [
        ("blank.ini", None),
        ("HOH.ini", None),
    ]
    theory_1 = os.path.join(prefix, "test_files", "theory_1.ini")
    theory_2 = os.path.join(prefix, "test_files", "theory_2.ini")

    def test_init(self):
        for i, (config_name, config) in enumerate(TestConfig.config_list):
            config = Config(
                os.path.join(prefix, "test_files", config_name),
                quiet=True,
                skip_user_default=True,
            )
            TestConfig.config_list[i] = config_name, config

            test = config.as_dict(skip=self.USER_SPECIFIC)
            ref_name = os.path.join(
                prefix, "ref_files", config_name.replace(".ini", "_init.json")
            )
            # with open(ref_name, "w") as f:
            #     json.dump(test, f, indent=2)

            # need this to make sure python->json stuff is consistent,
            # eg: json doesn't distinguish between tuples and lists
            test = json.loads(json.dumps(test))
            self.LOG.debug(json.dumps(test, indent=2))
            with open(ref_name, "r") as f:
                ref = json.load(f)
            self.maxDiff = None
            self.assertDictEqual(ref, test, msg=config_name)

    def test_parse_changes(self):
        ###
        # substitutions with single point of connections
        ###
        simple = Config(
            os.path.join(prefix, "test_files", "simple_subs.ini"),
            quiet=True,
            skip_user_default=True,
        )
        test = json.loads(json.dumps(simple._changes))
        ref_name = simple.infile.replace(".ini", "_changes.json")
        ref_name = ref_name.replace("test_files", "ref_files")
        # with open(ref_name, "w") as f:
        #     json.dump(simple._changes, f, indent=2)
        with open(ref_name, "r") as f:
            ref = json.load(f)
        self.assertDictEqual(ref, test, msg=simple.infile)

        ###
        # for ring-fusing substitutions (multiple connection points)
        ###
        rings = Config(
            os.path.join(prefix, "test_files", "ring_subs.ini"),
            quiet=True,
            skip_user_default=True,
        )
        test = json.loads(json.dumps(rings._changes))
        ref_name = rings.infile.replace(".ini", "_changes.json")
        ref_name = ref_name.replace("test_files", "ref_files")
        # with open(ref_name, "w") as f:
        #     json.dump(rings._changes, f, indent=2)
        with open(ref_name, "r") as f:
            ref = json.load(f)
        self.assertDictEqual(ref, test, msg=simple.infile)

    def test_for_loop(self):
        for_loop = Config(
            os.path.join(prefix, "test_files", "for_loop.ini"),
            quiet=True,
            skip_user_default=True,
        )
        structure_list = for_loop.get_template()
        for structure, kind in structure_list:
            dihedral = round(
                np.rad2deg(structure.dihedral("3", "1", "2", "6")), 6
            )
            if dihedral < 0:
                dihedral += 360
            self.assertEqual(float(structure.name.split(".")[-1]), dihedral)

    def test_call_on_suffix(self):
        config = Config(
            os.path.join(prefix, "test_files", "structure_suffix.ini"),
            quiet=True,
            skip_user_default=True,
        )

        structure_list = config.get_template()
        # for structure, kind in structure_list:
        #     print(structure, kind)

    def test_theory(self):
        config = Config(
            TestConfig.theory_1,
            quiet=True,
            skip_user_default=True,
        )
        geoms = config.get_template()
        geom = geoms[0][0]
        ref = Theory(
            processors=14,
            memory=28,
            method="b3lyp",
            basis="6-31G",
            solvent=ImplicitSolvent("PCM", "water"),
            empirical_dispersion="D3",
            job_type=OptimizationJob(geometry=geom),
            grid="(99, 590)",
            geometry=geom,
        )
        test = config.get_theory(geom)
        # print("\n", geom.write(outfile=False, theory=ref, style="gaussian"))
        
        self.assertEqual(ref, test)
        
        
        config = Config(
            TestConfig.theory_2,
            quiet=True,
            skip_user_default=True,
        )
        geoms = config.get_template()
        geom = geoms[0][0]
        ref = Theory(
            processors=14,
            memory=28,
            method="b3lyp",
            basis="zora-def2-tzvp aux J sarc Pt sarc-zora-tzvp",
            job_type="energy",
            geometry=geom,
            simple=["zora"],
        )
        # print("\n", geom.write(outfile=False, theory=ref, style="orca"))
        test = config.get_theory(geom)
        self.assertEqual(ref, test)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestConfig("test_init"))
    suite.addTest(TestConfig("test_parse_changes"))
    suite.addTest(TestConfig("test_for_loop"))
    suite.addTest(TestConfig("test_call_on_suffix"))
    suite.addTest(TestConfig("test_theory"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
