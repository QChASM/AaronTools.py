#!/usr/bin/env python3
import json
import os
import unittest

import AaronTools
import AaronTools.config
import numpy as np
from AaronTools.config import Config
from AaronTools.test import TestWithTimer, prefix, rmsd_tol, validate

AaronTools.config.AARONLIB = os.path.join(prefix, "aaron_libs")


class TestConfig(TestWithTimer):
    config_list = [("config_blank.ini", None), ("config_HOH.ini", None)]

    def test_init(self):
        for i, (config_name, config) in enumerate(TestConfig.config_list):
            config = Config(
                os.path.join(prefix, "test_files", config_name), quiet=True
            )
            config.set(
                "DEFAULT",
                "name",
                os.path.relpath(
                    config.get("DEFAULT", "name"),
                    config.get("DEFAULT", "top_dir"),
                ),
            )
            config.set("DEFAULT", "top_dir", "")
            TestConfig.config_list[i] = config_name, config

            test = {}
            for section in config:
                test[section] = dict(config.items(section))
            ref_name = os.path.join(
                prefix, "ref_files", config_name.replace(".ini", "_init.json")
            )
            with open(ref_name, "w") as f:
                json.dump(test, f, indent=2)
            with open(ref_name, "r") as f:
                ref = json.load(f)
            self.assertDictEqual(ref, test, msg=config_name)

    def test_parse_changes(self):
        simple = Config(os.path.join(prefix, "test_files", "simple_subs.ini"))
        simple._parse_changes()
        print(simple._changes)
        rings = Config(os.path.join(prefix, "test_files", "ring_subs.ini"))
        rings._parse_changes()
        print(rings._changes)
        test_1 = Config(os.path.join(prefix, "test_files", "substyle_1.ini"))
        test_1._parse_changes()
        print(test_1._changes)

    def test_for_loop(self):
        for_loop = Config(os.path.join(prefix, "test_files", "for_loop.ini"))
        structure_list = for_loop.get_template()
        for structure, kind in structure_list:
            dihedral = round(
                np.rad2deg(structure.dihedral("3", "1", "2", "6")), 6
            )
            if dihedral < 0:
                dihedral += 360
            self.assertEqual(float(structure.name.split(".")[-1]), dihedral)


def suite():
    suite = unittest.TestSuite()
    # suite.addTest(TestConfig("test_init"))
    suite.addTest(TestConfig("test_parse_changes"))
    # suite.addTest(TestConfig("test_for_loop"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
