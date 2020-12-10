import time
import unittest
from os.path import dirname

import numpy as np
from AaronTools.geometry import Geometry

prefix = dirname(__file__)


def rmsd_tol(geom, superTight=False, superLoose=False):
    """
    Automatically determine a reasonable rmsd tolerance for the input
    geometry based on its size and number of atoms
    """
    tolerance = len(geom.atoms) ** (
        2 - int(superTight) + int(superLoose)
    ) * np.sqrt(np.finfo(float).eps)

    com = geom.COM()
    max_d = None
    for atom in geom.atoms:
        d = np.linalg.norm(atom.coords - com)
        if max_d is None or d > max_d:
            max_d = d

    tolerance *= max_d * (2 - int(superTight) + int(superLoose))
    tolerance = tolerance ** (2 / (4 - int(superTight) + int(superLoose)))
    return tolerance


def check_atom_list(ref, comp):
    rv = True
    for i, j in zip(ref, comp):
        rv &= i.__repr__() == j.__repr__()
    return rv


def validate(test, ref, thresh=None, heavy_only=False, sort=True, debug=False):
    """
    Validates `test` geometry against `ref` geometry
    Returns: True if validation passed, False if failed

    :test: the geometry to validate
    :ref: the reference geometry
    :thresh: the RMSD threshold
        if thresh is a number: use that as threshold
        if thresh is None: use rmsd_tol() to determine
        if thresh is "tight": use rmsd_tol(superTight=True)
        if thresh is "loose": use rmsd_tol(superLoose=True)
    :sort: allow canonical sorting of atoms
    :debug: print info useful for debugging
    """
    if debug:
        print(ref.write("ref", outfile=False))
        print(test.write("test", outfile=False))

    if thresh is None:
        thresh = rmsd_tol(ref)
    try:
        thresh = float(thresh)
    except ValueError:
        if thresh.lower() == "tight":
            thresh = rmsd_tol(ref, superTight=True)
        elif thresh.lower() == "loose":
            thresh = rmsd_tol(ref, superLoose=True)
        else:
            raise ValueError("Bad threshold provided")

    # elements should all be the same
    t_el = sorted([t.element for t in test.atoms])
    r_el = sorted([r.element for r in ref.atoms])
    if len(t_el) != len(r_el):
        if debug:
            print(
                "wrong number of atoms: {} (test) vs. {} (ref)".format(
                    len(t_el), len(r_el)
                )
            )
        return False

    for t, r in zip(t_el, r_el):
        if t != r:
            if debug:
                print("elements don't match")
            return False
    # and RMSD should be below a threshold
    rmsd = test.RMSD(
        ref, align=debug, heavy_only=heavy_only, sort=sort, debug=debug
    )
    if debug:
        print("RMSD:", rmsd[2], "\tTHRESH:", thresh)
        rmsd[0].write("ref")
        rmsd[1].write("test")
        rmsd = rmsd[2]

    return rmsd < thresh


class TestWithTimer(unittest.TestCase):
    count = 0
    total_time = 0
    last_class = None
    last_result = None
    _errors = 0
    _fails = 0

    @classmethod
    def tearDownClass(cls):
        errors = len(cls.last_result.errors)
        fails = len(cls.last_result.failures)
        if TestWithTimer._errors != errors:
            TestWithTimer._errors = errors
            print("\b ERROR")
        elif TestWithTimer._fails != fails:
            TestWithTimer._fails = fails
            print("\b FAIL")
        else:
            print("\b ok")

        print(
            "Ran {} tests in {:.3f}s".format(
                TestWithTimer.count, TestWithTimer.total_time
            )
        )
        print("-" * 70)

    def setUp(self):
        errors = len(self._outcome.result.errors)
        fails = len(self._outcome.result.failures)
        if TestWithTimer._errors != errors:
            TestWithTimer._errors = errors
            print("\b ERROR")
        elif TestWithTimer._fails != fails:
            TestWithTimer._fails = fails
            print("\b FAIL")
        elif self.id().split(".")[1] == TestWithTimer.last_class:
            print("\b ok")
        elif TestWithTimer.last_class is not None:
            TestWithTimer.total_time = 0
        self.start_time = time.time()

    def tearDown(self):
        t = time.time() - self.start_time
        TestWithTimer.total_time += t

        name = self.id().split(".")[-2:]
        if not TestWithTimer.last_class or TestWithTimer.last_class != name[0]:
            TestWithTimer.last_class = name[0]
            TestWithTimer.count = 0
            print("\r{}:".format(name[0]))

        name = name[1]
        TestWithTimer.count += 1
        print(
            "\r    {:3.0f}: {:<30s} {: 3.3f}s  ".format(
                TestWithTimer.count, name, t
            ),
            end="",
        )

        TestWithTimer.last_result = self._outcome.result
