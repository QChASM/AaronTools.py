#! /usr/bin/env python3
import os
import unittest
from copy import copy

from AaronTools.comp_output import CompOutput
from AaronTools.test import TestWithTimer, prefix


class TestCompOutput(TestWithTimer):
    # with frequencies
    normal = CompOutput(os.path.join(prefix, "test_files", "normal.log"))
    died = CompOutput(os.path.join(prefix, "test_files", "died.log"))
    error = CompOutput(os.path.join(prefix, "test_files", "error.log"))
    # optimization
    opt_run = CompOutput(os.path.join(prefix, "test_files", "opt_running.log"))
    opt_norm = CompOutput(os.path.join(prefix, "test_files", "opt_normal.log"))

    def test_get_progress(self):
        test = [TestCompOutput.died.get_progress()]
        test += [TestCompOutput.error.get_progress()]
        test += [TestCompOutput.normal.get_progress()]
        test += [TestCompOutput.opt_run.get_progress()]

        ref = ["Progress not found"]
        ref += [
            "Max Force:0.003790/NO  RMS Force:0.000887/NO   Max Disp:1.095802/NO   RMS Disp:0.158612/NO"
        ]
        ref += [
            "Max Force:0.000016/YES RMS Force:0.000006/YES  Max Disp:0.011148/NO   RMS Disp:0.003723/NO"
        ]
        ref += [
            "Max Force:3.606006/NO  RMS Force:0.254588/NO   Max Disp:1.656082/NO   RMS Disp:0.279091/NO"
        ]

        for t, r in zip(test, ref):
            self.assertTrue(t == r)

    def test_grab_thermo(self):
        logs = [
            TestCompOutput.normal,
            TestCompOutput.error,
            TestCompOutput.opt_norm,
        ]

        tmp = logs[0]
        test = [
            [
                copy(tmp.energy),
                copy(tmp.enthalpy),
                copy(tmp.free_energy),
                copy(tmp.calc_Grimme_G()),
                copy(tmp.calc_G_corr(method="QHARM")),
            ]
        ]
        ref = [
            [
                -1856.01865834,
                -1855.440611,
                -1855.538011,
                -1855.5328046892625,
                0.486148,
            ]
        ]

        tmp = logs[1]
        test += [
            [copy(tmp.energy), copy(tmp.enthalpy), copy(tmp.free_energy), None]
        ]
        ref += [[1.00054354503, None, None, None]]

        tmp = logs[2]
        test += [
            [copy(tmp.energy), copy(tmp.enthalpy), copy(tmp.free_energy), None]
        ]
        ref += [[-3511.5160178, None, None, None]]

        for i, r in enumerate(ref):
            if r[3] is None:
                self.assertRaises(AttributeError, logs[i].calc_Grimme_G)
            for j in range(3):
                if r[j] is None:
                    self.assertTrue(test[i][j] is None)
                else:
                    self.assertTrue(test[i][j] == r[j])

    def test_crest(self):
        test = CompOutput(
            os.path.join(prefix, "test_files", "good.crest"), conf_name=False
        )


if __name__ == "__main__":
    unittest.main()
