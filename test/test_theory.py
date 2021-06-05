#!/usr/bin/env python3
import os
import unittest

from AaronTools import addlogger
from AaronTools.config import Config
from AaronTools.test import TestWithTimer, prefix


@addlogger
class TestTheory(TestWithTimer):
    LOG = None
    USER_SPECIFIC = ["metadata", "infile", (".*", ".*_dir")]

    def test_xtb_cmdline(self):
        config = Config(
            os.path.join(prefix, "test_files", "xtb_config.ini"),
            skip_user_default=True,
        )
        config = config.for_change("OH_Cl")
        structure = config.get_template()[0][0]
        structure = config.make_changes(structure)

        this_config = config.for_step(1)
        theory = this_config.get_theory(structure)
        ref = {
            "--opt": None,
            "--uhf": 1,
            "--etemp": "400",
            "--alpb": "acetone",
            "--restart": None,
            "-P": "2",
        }
        test = theory.get_xtb_cmdline(this_config)
        self.assertDictEqual(test, ref)

        this_config = config.for_step(2)
        theory = this_config.get_theory(structure)
        ref = {
            "--optts": None,
            "--uhf": 1,
            "--etemp": "400",
            "--gbsa": "acetone",
        }
        test = theory.get_xtb_cmdline(this_config)
        self.assertDictEqual(test, ref)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestTheory("test_xtb_cmdline"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
