#!/usr/bin/env python3
import json
import os
import unittest

import AaronTools
import AaronTools.config
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
            if i == 2:
                with open(ref_name, "w") as f:
                    json.dump(test, f, indent=2)
                print(json.dumps(test, indent=2))
            with open(ref_name, "r") as f:
                ref = json.load(f)
            self.assertDictEqual(ref, test, msg=config_name)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestConfig("test_init"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
