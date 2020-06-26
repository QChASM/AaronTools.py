"""For parsing and storing atom information"""
import json
import os
from copy import deepcopy
from warnings import warn

import numpy as np

from AaronTools.const import (
    CONNECTIVITY,
    EIJ,
    ELEMENTS,
    MASS,
    RADII,
    RIJ,
    SATURATION,
    TMETAL,
)

warn_LJ = set([])


class BondOrder:
    bonds = {}
    warn_atoms = set([])
    warn_str = (
        "\n"
        + "  Could not get bond order for: "
        + "    {} {}"
        + "    using bond order of 1"
    )
    # warn(s.format(a1.element, a2.element))

    def __init__(self):
        if BondOrder.bonds:
            return
        with open(
            os.path.join(
                os.path.dirname(__file__), "calculated_bond_lengths.json"
            )
        ) as f:
            BondOrder.bonds = json.load(f)

    @classmethod
    def key(cls, a1, a2):
        if isinstance(a1, Atom):
            a1 = a1.element
        if isinstance(a2, Atom):
            a2 = a2.element

        return " ".join(sorted([a1, a2]))

    @classmethod
    def get(cls, a1, a2):
        """determines bond order between two atoms based on bond length"""
        try:
            bonds = cls.bonds[cls.key(a1, a2)]
        except KeyError:
            if a1.element == "H" or a2.element == "H":
                return 1
            else:
                BondOrder.warn_atoms.add((a1.element, a2.element))
                return 1
        dist = a1.dist(a2)
        closest = 0, None  # (bond order, length diff)
        for order, length in bonds.items():
            diff = abs(length - dist)
            if closest[1] is None or diff < closest[1]:
                closest = order, diff
        return float(closest[0])


class Atom:
    """
    Attributes:
        element         str
        coords          np.array(float)
        flag            bool            true if frozen, false if relaxed
        name            str             form of \d+(\.\d+)*
        tags            set
        connected       set(Atom)
        constraint      set(Atom)       for determining constrained bonds
        _rank
        _radii          float           for calculating if bonded
        _connectivity   int             max connections without hypervalence
        _saturation     int             max connections without hypervalence or charges
    """

    BondOrder()

    def __init__(self, element="", coords=[], flag=False, name="", tags=[]):
        element = str(element).strip().capitalize()
        if element == "":
            self.element = element
            self._radii = None
            self._connectivity = None
        elif element in ELEMENTS:
            self.element = element
            self._set_radii()
            self._set_connectivity()
            self._set_saturation()
        else:
            raise ValueError("Unknown element detected:", element)

        self.coords = np.array(coords, dtype=float)
        self.flag = bool(flag)
        self.name = str(name).strip()

        if hasattr(tags, "__iter__") and not isinstance(tags, str):
            self.tags = set(tags)
        else:
            self.tags = set([tags])

        self.connected = set([])
        self.constraint = set([])
        self._rank = None

    # utilities
    def __float__(self):
        """
        converts self.name from a string to a floating point number
        """
        rv = self.name.split(".")
        if len(rv) == 0:
            return float(0)
        if len(rv) == 1:
            return float(rv[0])
        rv = "{}.{}".format(rv[0], rv[1])
        return float(rv)

    def __lt__(self, other):
        """
        sorts by canonical smiles rank, then by invariant, then by name
            more connections first
            then, more non-H bonds first
            then, higher atomic number first
            then, higher number of attached hydrogens first
            then, lower sorting name first
        """
        if (
            self._rank is not None
            and other._rank is not None
            and self._rank != other._rank
        ):
            return self._rank > other._rank

        a = self.get_invariant()
        b = other.get_invariant()
        if a != b:
            return a > b

        a = self.name.split(".")
        b = other.name.split(".")
        while len(a) < len(b):
            a += ["0"]
        while len(b) < len(a):
            b += ["0"]
        for i, j in zip(a, b):
            try:
                if int(i) != int(j):
                    return int(i) < int(j)
            except ValueError:
                pass
        return True

    def __repr__(self):
        s = ""
        s += "{:>3s}  ".format(self.element)
        for c in self.coords:
            s += "{: 13.8f} ".format(c)
        s += " {: 2d}  ".format(-1 if self.flag else 0)
        s += "{}".format(self.name)
        return s

    def _set_radii(self):
        """Sets atomic radii"""
        try:
            self._radii = float(RADII[self.element])
        except KeyError:
            warn("Radii not found for element: %s" % self.element)
        return

    def _set_connectivity(self):
        """Sets theoretical maximum connectivity.
        If # connections > self._connectivity, then atom is hyper-valent
        """
        try:
            self._connectivity = int(CONNECTIVITY[self.element])
        except KeyError:
            warn("Connectivity not found for element: " + self.element)
        return

    def _set_saturation(self):
        """Sets theoretical maximum connectivity without the atom having a formal charge.
        If # connections > self._saturation, then atom is hyper-valent or has a non-zero formal charge
        """
        try:
            self._saturation = int(SATURATION[self.element])
        except KeyError:
            if self.element not in TMETAL:
                warn("Saturation not found for element: " + self.element)
        return

    def add_tag(self, *args):
        for a in args:
            if hasattr(a, "__iter__") and not isinstance(a, str):
                self.tags = self.tags.union(set(a))
            else:
                self.tags.add(a)
        return

    def get_invariant(self):
        """gets initial invariant
        (1) number of non-hydrogen connections (\d{1}): nconn
        (2) sum of bond order of non-hydrogen bonds * 10 (\d{2}): nB
        (3) atomic number (\d{3}): z
        #(4) sign of charge (\d{1})
        #(5) absolute charge (\d{1})
        (6) number of attached hydrogens (\d{1}): nH
        """
        heavy = set([x for x in self.connected if x.element != "H"])
        # number of non-hydrogen connections:
        nconn = len(heavy)
        # number of bonds with heavy atoms
        nB = 0
        for h in heavy:
            nB += BondOrder.get(h, self)
        # number of connected hydrogens
        nH = len(self.connected - heavy)
        # atomic number
        z = ELEMENTS.index(self.element)

        return "{:01d}{:03d}{:03d}{:01d}".format(
            int(nconn), int(nB * 10), int(z), int(nH)
        )

    def copy(self):
        rv = Atom()
        for key, val in self.__dict__.items():
            if key == "connected":
                continue
            if key == "constraint":
                continue
            try:
                rv.__dict__[key] = val.copy()
            except AttributeError:
                rv.__dict__[key] = val
                if val.__class__.__module__ != "builtins":
                    warn(
                        "No copy method for {}: in-place changes may occur".format(
                            type(val)
                        )
                    )
        return rv

    # measurement
    def is_connected(self, other, tolerance=None):
        """determines if distance between atoms is small enough to be bonded"""
        if tolerance is None:
            tolerance = 0.3

        if self._radii is None:
            self._set_radii()
        if other._radii is None:
            other._set_radii()
        cutoff = self._radii + other._radii + tolerance
        return self.dist(other) < cutoff

    def bond(self, other):
        """returns the vector self-->other"""
        return other.coords - self.coords

    def dist(self, other):
        """returns the distance between self and other"""
        return np.linalg.norm(self.bond(other))

    def angle(self, a1, a3):
        """returns the a1-self-a3 angle"""
        v1 = self.bond(a1)
        v2 = self.bond(a3)
        dot = np.dot(v1, v2)
        return np.arccos(dot / (self.dist(a1) * self.dist(a3)))

    def mass(self):
        """returns atomic mass"""
        return MASS[self.element]

    def rij(self, other):
        try:
            rv = RIJ[self.element + other.element]
        except KeyError:
            try:
                rv = RIJ[other.element + self.element]
            except KeyError:
                warn_LJ.add("".join(sorted([self.element, other.element])))
                return 0
        return rv

    def eij(self, other):
        try:
            rv = EIJ[self.element + other.element]
        except KeyError:
            try:
                rv = EIJ[other.element + self.element]
            except KeyError:
                warn_LJ.add("".join(sorted([self.element, other.element])))
                return 0
        return rv

    def bond_order(self, other):
        return BondOrder.get(self, other)

    @classmethod
    def get_shape(cls, shape_name):
        """returns dummy atoms in an idealized vsepr geometry
        shape_name can be:
        linear 1
        linear 2
        bent 2a (tetrahedral electron geometry w/ 2 bonds)
        bent 2b (trigonal planar electron geometry)
        trigonal planar
        bent 3 (tetrahedral electron geometry w/ 3 bonds)
        t shaped
        sawhorse
        square planar
        trigonal bipyramid
        square pyramid
        octahedral
        """
        if shape_name == "linear 1":
            return cls.linear_shape()[0:2]
        elif shape_name == "linear 2":
            return cls.linear_shape()
        elif shape_name == "bent 2a":
            return cls.tetrahedral_shape()[0:3]
        elif shape_name == "bent 2b":
            return cls.trigonal_planar_shape()[0:3]
        elif shape_name == "trigonal planar":
            return cls.trigonal_planar_shape()
        elif shape_name == "bent 3":
            return cls.tetrahedral_shape()[0:4]
        elif shape_name == "t shaped":
            return cls.octahedral_shape()[0:4]
        elif shape_name == "tetrahedral":
            return cls.tetrahedral_shape()
        elif shape_name == "sawhorse":
            return cls.trigonal_bipyramidal_shape()[0:5]
        elif shape_name == "square planar":
            return cls.octahedral_shape()[0:5]
        elif shape_name == "trigonal bipyramid":
            return cls.trigonal_bipyramid_shape()
        elif shape_name == "square pyramid":
            return cls.octahedral_shape()[0:6]
        elif shape_name == "octahedral":
            return cls.octahedral_shape()
        else:
            raise RuntimeError(
                "no shape method is defined for %s" % shape_name
            )

    @classmethod
    def linear_shape(cls):
        center = Atom("X", np.zeros(3), name="0")
        pos1 = Atom("X", np.array([1.0, 0.0, 0.0]), name="1")
        pos2 = Atom("X", np.array([-1.0, 0.0, 0.0]), name="2")

        return [center, pos1, pos2]

    @classmethod
    def trigonal_planar_shape(cls):
        positions = cls.trigonal_bipyramidal_shape()
        return positions[:-2]

    @classmethod
    def tetrahedral_shape(cls):
        center = Atom("X", np.zeros(3), name="0")
        angle = np.deg2rad(109.471 / 2)
        pos1 = Atom(
            "X", np.array([np.cos(angle), -np.sin(angle), 0.0]), name="1"
        )
        pos2 = Atom(
            "X", np.array([np.cos(angle), np.sin(angle), 0.0]), name="2"
        )
        pos3 = Atom(
            "X", np.array([-np.cos(angle), 0.0, np.sin(angle)]), name="3"
        )
        pos4 = Atom(
            "X", np.array([-np.cos(angle), 0.0, -np.sin(angle)]), name="4"
        )

        return [center, pos1, pos2, pos3, pos4]

    @classmethod
    def trigonal_bipyramidal_shape(cls):
        center = Atom("X", np.zeros(3), name="0")
        angle = np.deg2rad(120)
        pos1 = Atom("X", np.array([1.0, 0.0, 0.0]), name="1")
        pos2 = Atom(
            "X", np.array([np.cos(angle), np.sin(angle), 0.0]), name="2"
        )
        pos3 = Atom(
            "X", np.array([np.cos(angle), -np.sin(angle), 0.0]), name="3"
        )
        pos4 = Atom("X", np.array([0.0, 0.0, 1.0]), name="4")
        pos5 = Atom("X", np.array([0.0, 0.0, -1.0]), name="5")

        return [center, pos1, pos2, pos3, pos4, pos5]

    @classmethod
    def octahedral_shape(cls):
        center = Atom("X", np.zeros(3), name="0")
        pos1 = Atom("X", np.array([1.0, 0.0, 0.0]), name="1")
        pos2 = Atom("X", np.array([-1.0, 0.0, 0.0]), name="2")
        pos3 = Atom("X", np.array([0.0, 1.0, 0.0]), name="3")
        pos4 = Atom("X", np.array([0.0, -1.0, 0.0]), name="4")
        pos5 = Atom("X", np.array([0.0, 0.0, 1.0]), name="5")
        pos6 = Atom("X", np.array([0.0, 0.0, -1.0]), name="6")

        return [center, pos1, pos2, pos3, pos4, pos5, pos6]

    @staticmethod
    def new_shape(old_shape, new_connectivity, bond_change):
        """EXPERIMENTAL
        returns the name of the expected vsepr geometry when the number of bonds
        changes by +/- 1

        old_shape - :str: vsepr geometry name
        new_connectivity - :int: connectivity (see Atom._connectivity)
        bond_change - :int: +1 or -1, indicating that the number of bonds is changing by 1"""
        if old_shape == "linear 1":
            if bond_change == 1:
                return "linear 2"
            elif bond_change == -1:
                return None

        elif old_shape == "linear 2":
            if bond_change == 1:
                if new_connectivity is not None and new_connectivity > 4:
                    return "t shaped"
                else:
                    return "trigonal planar"
            elif bond_change == -1:
                return "linear 1"

        elif old_shape == "bent 2a":
            if bond_change == 1:
                return "bent 3"
            elif bond_change == -1:
                return "linear 1"

        elif old_shape == "bent 2b":
            if bond_change == 1:
                return "trigonal planar"
            elif bond_change == -1:
                return "linear 1"

        elif old_shape == "trigonal planar":
            if bond_change == 1:
                return "tetrahedral"
            elif bond_change == -1:
                if new_connectivity == 4:
                    return "bent 2a"
                else:
                    return "bent 2b"

        elif old_shape == "bent 3":
            if bond_change == 1:
                return "tetrahedral"
            elif bond_change == -1:
                return "bent 2a"

        elif old_shape == "t shaped":
            if bond_change == 1:
                if new_connectivity == 6:
                    return "square planar"
                else:
                    return "sawhorse"
            elif bond_change == -1:
                return "linear 2"

        elif old_shape == "tetrahedral":
            if bond_change == 1:
                return "trigonal bipyramid"
            elif bond_change == -1:
                return "bent 3"

        elif old_shape == "square planar":
            if bond_change == 1:
                return "trigonal bipyramid"
            elif bond_change == -1:
                return "t shaped"

        elif old_shape == "trigonal bipyramid":
            if bond_change == 1:
                return "octahedral"
            elif bond_change == -1:
                return "sawhorse"

        elif old_shape == "octahedral":
            if bond_change == -1:
                return "trigonal bipryamid"

        else:
            raise RuntimeError("no shape method is defined for %s" % old_shape)

    def get_vsepr(self):
        """determine vsepr geometry around an atom
        e.g. tetrahedral, sawhorse, etc.
        returns geometry name as a string and the score assigned to that geometry
        scores > 0.5 are generally questionable"""

        # determine what geometries to try based on the number of bonded atoms
        try_shapes = {}
        if len(self.connected) == 1:
            try_shapes["linear 1"] = Atom.get_shape("linear 1")

        elif len(self.connected) == 2:
            try_shapes["linear 2"] = Atom.get_shape("linear 2")
            try_shapes["bent 2a"] = Atom.get_shape("bent 2a")
            try_shapes["bent 2b"] = Atom.get_shape("bent 2b")

        elif len(self.connected) == 3:
            try_shapes["trigonal planar"] = Atom.get_shape("trigonal planar")
            try_shapes["bent 3"] = Atom.get_shape("bent 3")
            try_shapes["t shaped"] = Atom.get_shape("t shaped")

        elif len(self.connected) == 4:
            try_shapes["tetrahedral"] = Atom.get_shape("tetrahedral")
            try_shapes["sawhorse"] = Atom.get_shape("sawhorse")
            try_shapes["square planar"] = Atom.get_shape("square planar")

        elif len(self.connected) == 5:
            try_shapes["trigonal bipyramid"] = Atom.get_shape(
                "trigonal bipyramid"
            )
            try_shapes["square pyramid"] = Atom.get_shape("square pyramid")

        elif len(self.connected) == 6:
            try_shapes["octahedral"] = Atom.get_shape("octahedral")

        else:
            return None

        # make a copy of the atom and the atoms bonded to it
        # set each bond length to 1 to more easily compare to the
        # idealized shapes from Atom
        adjusted_shape = [deepcopy(atom) for atom in [self, *self.connected]]
        for atom in adjusted_shape:
            atom.coords -= self.coords

        for atom in adjusted_shape[1:]:
            atom.coords /= atom.dist(adjusted_shape[0])

        Y = np.array([position.coords for position in adjusted_shape])
        r1 = np.matmul(np.transpose(Y), Y)
        u, s1, vh = np.linalg.svd(r1)

        best_score = None
        best_shape = None
        for shape in try_shapes:
            X = np.array([position.coords for position in try_shapes[shape]])
            r2 = np.matmul(np.transpose(X), X)
            u, s2, vh = np.linalg.svd(r2)

            score = sum([abs(x1 - x2) for x1, x2 in zip(s1, s2)])
            if best_score is None or score < best_score:
                best_score = score
                best_shape = shape

        return best_shape, best_score