#!/usr/bin/env python3

import json
import os
import re
from copy import deepcopy
from glob import glob
from warnings import warn

import numpy as np

from AaronTools.const import AARONLIB, QCHASM
from AaronTools.fileIO import FileReader
from AaronTools.geometry import Geometry


class Substituent(Geometry):
    """
    Attributes:
        name
        atoms
        end         the atom substituent is connected to
        conf_num    number of conformers
        conf_angle  angle to rotate by to make next conformer
    """

    AARON_LIBS = os.path.join(AARONLIB, "Subs/*.xyz")
    BUILTIN = os.path.join(QCHASM, "AaronTools/Substituents/*.xyz")
    CACHE_FILE = os.path.join(os.path.dirname(__file__), "cache/substituents")

    try:
        with open(CACHE_FILE) as f:
            cache = json.load(f)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        cache = {}
        cache["lengths"] = {}  # for storing number of atoms in each sub

    def __init__(
        self,
        sub,
        name=None,
        targets=None,
        end=None,
        conf_num=None,
        conf_angle=None,
    ):
        """
        sub is either a file sub, a geometry, or an atom list
        """
        if isinstance(sub, (Geometry, list)):
            # we can create substituent object from fragment
            if isinstance(sub, Substituent):
                self.name = name if name else sub.name
                self.conf_angle = conf_num if conf_num else sub.conf_angle
                self.conf_angle = conf_angle if conf_angle else sub.conf_angle
            elif isinstance(sub, Geometry):
                self.name = name if name else sub.name
                self.conf_angle = conf_num
                self.conf_angle = conf_angle
            else:
                self.name = name
                self.conf_num = conf_num
                self.conf_angle = conf_angle

            # save atom info
            if targets is None:
                try:
                    self.atoms = sub.atoms
                except AttributeError:
                    self.atoms = sub
            else:
                self.atoms = sub.find(targets)
            self.refresh_connected(rank=False)

            # detect sub and conformer info
            if not conf_num or not conf_angle:
                if not self.detect_sub():
                    LookupError(
                        "Substituent not found in library: " + str(self.name)
                    )
        else:  # or we can create from file
            # find substituent xyz file
            fsub = None
            for f in glob(Substituent.AARON_LIBS) + glob(Substituent.BUILTIN):
                match = re.search("/" + sub + ".xyz", f)
                if match is not None:
                    fsub = f
                    break
            # or assume we were given a file name instead
            if not fsub and ".xyz" in sub:
                fsub = sub
                sub = sub.split("/")[-1].rstrip(".xyz")

            # load in atom info
            from_file = FileReader(fsub)
            self.name = sub
            self.comment = from_file.comment
            self.atoms = from_file.atoms
            if targets is not None:
                self.atoms = self.find(targets)
            self.refresh_connected(rank=False)

            # set conformer info
            conf_info = re.search("CF:(\d+),(\d+)", self.comment)
            if conf_info is not None:
                self.conf_num = int(conf_info.group(1))
                self.conf_angle = np.deg2rad(float(conf_info.group(2)))
            else:
                warn("Conformer info not loaded for" + f)

        # end is connection point
        self.end = end
        if not self.name:
            self.name = "sub"
        if self.name == "sub" and end is not None:
            self.name += "-{}".format(end.name)

    def __lt__(self, other):
        if len(self.atoms) != len(other.atoms):
            return self.atoms < other.atoms
        elif self.end != other.end:
            return self.end < other.end
        else:
            return self.atoms[0] < other.atoms[0]

    def copy(self, atoms=None, name=None, targets=None, end=None):
        """
        creates a new copy of the geometry
        parameters:
            atoms (list): defaults to all atoms
            name (str): defaults to NAME_copy
        """
        if atoms is None:
            atoms = deepcopy(self.atoms)
        if name is None:
            name = self.name
        if end is None:
            end = self.end
        return Substituent(atoms, name, targets=targets, end=end)

    def detect_sub(self):
        """
        detects conformer information for a substituent by searching the
        substituent library
        """
        sub_lengths = Substituent.cache["lengths"]
        found = False
        cache_changed = False

        for f in glob(Substituent.AARON_LIBS) + glob(Substituent.BUILTIN):
            match = re.search("/([^/]*).xyz", f)
            name = match.group(1)
            # test number of atoms against cache
            if name in sub_lengths and len(self.atoms) != sub_lengths[name]:
                continue
            test_sub = Substituent(name)
            # add to cache
            sub_lengths[test_sub.name] = len(test_sub.atoms)
            cache_changed = True

            # want same number of atoms
            if len(self.atoms) != len(test_sub.atoms):
                continue

            bad = False
            for i, j in zip(sorted(self.atoms), sorted(test_sub.atoms)):
                # and correct elements
                if i.element != j.element:
                    bad = True
                    break
                # and correct connections
                if len(i.connected) != len(j.connected):
                    bad = True
                    break
                for ii, jj in zip(sorted(i.connected), sorted(j.connected)):
                    if ii.element != jj.element:
                        bad = True
                        break
                if bad:
                    break
            if bad:
                continue

            # if found, save name and conf info
            self.name = test_sub.name
            self.comment = test_sub.comment
            self.conf_angle = test_sub.conf_angle
            self.conf_num = test_sub.conf_num
            found = True

        # update cache
        if cache_changed:
            Substituent.cache["lengths"] = sub_lengths
            with open(Substituent.CACHE_FILE, "w") as f:
                json.dump(Substituent.cache, f)
        return found

    def align_to_bond(self, bond):
        """
        align substituent to a bond vector
        """
        bond /= np.linalg.norm(bond)
        x_axis = np.array([1.0, 0.0, 0.0])
        rot_axis = np.cross(x_axis, bond)
        rot_axis /= np.linalg.norm(rot_axis)
        angle = np.arccos(np.dot(bond, x_axis))
        self.rotate(rot_axis, angle)

    def sub_rotate(self, angle=None):
        """
        rotates substituent about bond w/ rest of geometry
        :angle: in radians
        """
        if angle is None:
            angle = self.conf_angle
        axis = self.atoms[0].bond(self.end)
        self.rotate(axis, angle, center=self.end)
