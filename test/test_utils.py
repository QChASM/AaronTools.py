#! /usr/bin/env python3
import os
import unittest

from AaronTools.test import TestWithTimer, prefix, rmsd_tol, validate
from AaronTools.utils import utils


class TestCatalyst(TestWithTimer):
    def test_same_cycle(self):
        graph = [[1, 2], [0, 2], [0, 1, 3], [2]]
        print(graph)
        self.assertFalse(utils.same_cycle(graph, 2, 3))
        print()
        print(graph)
        self.assertFalse(utils.same_cycle(graph, 0, 3))
        print()
        print(graph)
        self.assertTrue(utils.same_cycle(graph, 0, 1))
        print()

        graph = [[1, 2, 3], [0, 2], [0, 1], [0, 4], [3]]
        print(graph)
        self.assertFalse(utils.same_cycle(graph, 2, 3))
        print()
        print(graph)
        self.assertFalse(utils.same_cycle(graph, 0, 4))
        print()
        print(graph)
        self.assertTrue(utils.same_cycle(graph, 1, 0))
        print()
        print(graph)
        self.assertTrue(utils.same_cycle(graph, 2, 0))
        print()


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestCatalyst("test_same_cycle"))
    return suite


if __name__ == "__main__":
    # for running specific tests, change test name in suite()
    runner = unittest.TextTestRunner()
    runner.run(suite())
