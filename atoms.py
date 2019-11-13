"""For parsing and storing atom information"""
import json
import os
from warnings import warn

import numpy as np

from AaronTools.const import CONNECTIVITY, EIJ, ELEMENTS, MASS, RADII, RIJ

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
        sorts by canonical smiles invariant, then by name
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
            if int(i) != int(j):
                return int(i) < int(j)
        else:
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
