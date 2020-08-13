#!/usr/bin/env python3

import json
import os
import re
from copy import deepcopy
from glob import glob
from warnings import warn

import numpy as np

from AaronTools.const import AARONLIB, QCHASM, VDW_RADII
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

    AARON_LIBS = os.path.join(AARONLIB, "Subs", "*.xyz")
    BUILTIN = os.path.join(QCHASM, "AaronTools", "Substituents", "*.xyz")
    CACHE_FILE = os.path.join(AARONLIB, "cache", "substituents.json")

    try:
        with open(CACHE_FILE) as f:
            cache = json.load(f)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        cache = {}
        cache["lengths"] = {}  # for storing number of atoms in each sub

    def __init__(
        self,
        sub=None,
        name=None,
        targets=None,
        end=None,
        conf_num=None,
        conf_angle=None,
        detect=True,
    ):
        """
        sub is either a file sub, a geometry, or an atom list
        """
        self.name = name
        self.atoms = []
        self.end = end
        self.conf_angle = conf_angle
        self.conf_num = conf_num
        self.comment = None
        if sub is None:
            return

        if isinstance(sub, (Geometry, list)):
            # we can create substituent object from fragment
            if isinstance(sub, Substituent):
                self.name = name if name else sub.name
                self.conf_num = conf_num if conf_num else sub.conf_num
                self.conf_angle = conf_angle if conf_angle else sub.conf_angle
                self.comment = sub.comment
            elif isinstance(sub, Geometry):
                self.name = name if name else sub.name
                self.conf_num = conf_num
                self.conf_angle = conf_angle
                self.comment = sub.comment
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

            # detect sub and conformer info
            if detect and (not conf_num or not conf_angle):
                if not self.detect_sub():
                    LookupError(
                        "Substituent not found in library: " + str(self.name)
                    )
        else:  # or we can create from file
            # find substituent xyz file
            fsub = None
            for f in glob(Substituent.AARON_LIBS) + glob(Substituent.BUILTIN):
                match = sub + ".xyz" == os.path.basename(f)
                if match:
                    fsub = f
                    break
            # or assume we were given a file name instead
            if not fsub and ".xyz" in sub:
                fsub = sub
                sub = os.path.basename(sub).rstrip(".xyz")

            if fsub is None:
                raise RuntimeError("substituent name not recognized: %s" % sub)

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

        if not self.name:
            self.name = "sub"
        if self.name == "sub" and end is not None:
            self.name += "-{}".format(end.name)

    def __lt__(self, other):
        if self.end < other.end and not other.end < self.end:
            return True
        if len(self.atoms) != len(other.atoms):
            return len(self.atoms) < len(other.atoms)
        for a, b in zip(
            self.reorder(start=self.atoms[0])[0],
            other.reorder(start=other.atoms[0])[0],
        ):
            if a < b and not b < a:
                return True
        return False

    def copy(self, end=None):
        """
        creates a new copy of the geometry
        parameters:
            atoms (list): defaults to all atoms
            name (str): defaults to NAME_copy
        """
        rv = super().copy()
        rv = Substituent(
            rv,
            end=end,
            conf_angle=self.conf_angle,
            conf_num=self.conf_num,
            detect=False,
        )
        if end is not None:
            rv.atoms[0].connected.add(rv.end)
        return rv

    @classmethod
    def list(cls):
        names = []
        for f in glob(cls.AARON_LIBS) + glob(cls.BUILTIN):
            name = os.path.splitext(os.path.basename(f))[0]
            names.append(name)

        return names

    def detect_sub(self):
        """
        detects conformer information for a substituent by searching the
        substituent library
        """
        sub_lengths = Substituent.cache["lengths"]
        found = False
        cache_changed = False

        # temporarily detach end from sub so the connectivity is same as
        # for the library substituent by itself
        test_sub = self.copy()
        test_sub.refresh_ranks()

        for f in glob(Substituent.AARON_LIBS) + glob(Substituent.BUILTIN):
            filename = os.path.basename(f)
            match = re.search("^([\s\S]+).xyz", filename)
            name = match.group(1)
            # test number of atoms against cache
            if (
                name in sub_lengths
                and len(test_sub.atoms) != sub_lengths[name]
            ):
                continue
            ref_sub = Substituent(name)
            ref_sub.refresh_ranks()
            # add to cache
            sub_lengths[ref_sub.name] = len(ref_sub.atoms)
            cache_changed = True

            # want same number of atoms
            if len(test_sub.atoms) != len(ref_sub.atoms):
                continue

            for a, b in zip(sorted(test_sub.atoms), sorted(ref_sub.atoms)):
                # want correct elements
                if a.element != b.element:
                    break
                # and correct connections
                if len(a.connected) != len(b.connected):
                    break
                # and correct connected elements
                failed = False
                for i, j in zip(
                    sorted([aa.element for aa in a.connected]),
                    sorted([bb.element for bb in b.connected]),
                ):
                    if i != j:
                        failed = True
                        break
                if failed:
                    break
            else:
                # if found, save name and conf info
                self.name = ref_sub.name
                self.comment = ref_sub.comment
                self.conf_angle = ref_sub.conf_angle
                self.conf_num = ref_sub.conf_num
                break

        # update cache
        if cache_changed:
            Substituent.cache["lengths"] = sub_lengths
            if not os.path.exists(os.path.dirname(Substituent.CACHE_FILE)):
                os.makedirs(os.path.dirname(Substituent.CACHE_FILE))

            with open(Substituent.CACHE_FILE, "w") as f:
                json.dump(Substituent.cache, f)

        return found

    def sterimol(self, parameter='L', return_vector=False):
        """returns sterimol parameter value for the specified parameter
        return_vector: bool/returns tuple(vector start, vector end) instead
        parameter (str) can be:
            'L'
            'B5'
        """
        if self.end is None:
            raise RuntimeError("cannot calculate sterimol values for substituents without end")

        from AaronTools.finders import BondedTo

        atom1 = self.find(BondedTo(self.end))[0]
        atom2 = self.end

        print(atom1.name, atom2.name)

        L_axis = atom2.bond(atom1)
        L_axis /= np.linalg.norm(L_axis)

        param_value = None
        vector = None
        
        for atom in self.atoms:
            test_v = atom2.bond(atom)
            test_L = np.dot(test_v, L_axis) - atom1.dist(atom2) + \
                     VDW_RADII[atom1.element] + VDW_RADII[atom.element]
            
            if parameter == 'L':
                test_L = np.dot(test_v, L_axis) - atom1.dist(atom2) + \
                         VDW_RADII[atom1.element] + VDW_RADII[atom.element]
                if param_value is None or test_L > param_value:
                    param_value = test_L
                    start = atom1.coords - VDW_RADII[atom1.element] * L_axis
                    vector = (start, start + param_value * L_axis)

            elif parameter == 'B1':
                b = np.dot(test_v, L_axis)
                test_B1_v = test_v - (b * L_axis)
                test_B1 = np.linalg.norm(test_B1_v) + VDW_RADII[atom.element]
                if (param_value is None or test_B1 < param_value) and \
                   (len(self.atoms) == 1 or atom is not atom1):
                    param_value = test_B1
                    start = atom.coords - test_B1_v
                    if np.linalg.norm(test_B1_v) > 3*np.finfo(float).eps:
                        perp_vec = test_B1_v
                    else:
                        v_n = test_v / np.linalg.norm(test_v)
                        perp_vec = v_n[::-1]
                        perp_vec -= np.dot(v_n, perp_vec) * v_n
                    
                    end = start + test_B1 * (perp_vec / np.linalg.norm(perp_vec))
                    
                    vector = (start, end)

            elif parameter == 'B5':
                b = np.dot(test_v, L_axis)
                test_B5_v = test_v - (b * L_axis)
                test_B5 = np.linalg.norm(test_B5_v) + VDW_RADII[atom.element]
                if param_value is None or test_B5 > param_value:
                    param_value = test_B5
                    start = atom.coords - test_B5_v
                    if np.linalg.norm(test_B5_v) > 3*np.finfo(float).eps:
                        perp_vec = test_B5_v
                    else:
                        v_n = test_v / np.linalg.norm(test_v)
                        perp_vec = v_n[::-1]
                        perp_vec -= np.dot(v_n, perp_vec) * v_n
                    
                    end = start + test_B5 * (perp_vec / np.linalg.norm(perp_vec))
                    
                    vector = (start, end)

        if return_vector:
            return vector
        else:
            return param_value
    
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

    def sub_rotate(self, angle=None, reverse=False):
        """
        rotates substituent about bond w/ rest of geometry
        :angle: in radians
        """
        if angle is None:
            angle = self.conf_angle
        if reverse:
            angle *= -1
        axis = self.atoms[0].bond(self.end)
        self.rotate(axis, angle, center=self.end)

    def rebuild(self):
        super().rebuild(start=self.atoms[0])
