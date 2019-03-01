"""For parsing and storing atom information"""
import json
from os import path
from warnings import warn
from collections import deque

import numpy as np
from AaronTools.const import RADII, ELEMENTS, MASS, RIJ, EIJ
from AaronTools.const import CONNECTIVITY, ELECTRONEGATIVITY


warn_LJ = set([])


class BondOrder:
    bonds = {}
    warn_atoms = set([])
    warn_str = "\n" + \
        "  Could not get bond order for: " + \
        "    {} {}" + \
        "    using bond order of 1"
    # warn(s.format(a1.element, a2.element))

    def __init__(self):
        with open(path.dirname(__file__) + "/bond_data.json") as f:
            bond_data = json.load(f)

        for b in bond_data:
            key = " ".join(sorted(b['atoms']))
            try:
                BondOrder.bonds[key] += [b]
            except KeyError:
                BondOrder.bonds[key] = [b]
        return

    @classmethod
    def key(cls, a1, a2):
        if isinstance(a1, Atom):
            a1 = a1.element
        if isinstance(a2, Atom):
            a2 = a2.element

        if a1 == a2:
            return a1
        return " ".join(sorted([a1, a2]))

    @classmethod
    def get(cls, a1, a2):
        """determines bond order between two atoms based on bond lenght"""
        try:
            bonds = cls.bonds[cls.key(a1, a2)]
        except KeyError:
            if a1.element == 'H' or a2.element == 'H':
                return 1
            else:
                BondOrder.warn_atoms.add((a1.element, a2.element))
                return 1
        dist = a1.dist(a2)
        closest = None
        for b in bonds:
            diff = abs(b['mean'] - dist)
            if closest is None or diff < closest[0]:
                closest = (diff, b)

        # if abs(diff - closest[1]['stdev']) > 3*closest[1]['stdev']:
            # s = """
            # Bond order prediction outside of three standard deviations
              # Atoms in question:
                # {}
                # {}
              # Predicted bond order: {}
            # """
            # warn(s.format(a1, a2, closest[1]['order']))
        return closest[1]['order']


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

    def __init__(self, element='', coords=[], flag=False, name='', tags=[]):
        element = str(element).strip().capitalize()
        if element == '':
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

        if hasattr(tags, '__iter__') and not isinstance(tags, str):
            self.tags = set(tags)
        else:
            self.tags = set([tags])

        self.connected = set([])
        self.constraint = set([])
        self._rank = None

        return

    # utilities
    def to_json(self):
        tmp = {}
        tmp['element'] = self.element
        tmp['coords'] = list(self.coords)
        tmp['flag'] = self.flag
        tmp['name'] = self.name
        tmp['tags'] = list(sorted(self.tags))
        return json.dumps(tmp)

    def from_json(self, parse):
        parse = json.loads(parse)
        for key, val in parse.items():
            if key == 'tags':
                self.__dict__[key] = set(val)
            elif key == 'coords':
                self.__dict__[key] = np.array(val)
            else:
                self.__dict__[key] = val
        return self

    def __float__(self):
        """
        converts self.name from a string to a floating point number
        """
        rv = self.name.split('.')
        if len(rv) == 0:
            return float(0)
        if len(rv) == 1:
            return float(rv[0])
        rv = "{}.{}".format(rv[0], rv[1])
        return float(rv)

    def __repr__(self):
        s = ''
        s += "{:>3s}  ".format(self.element)
        for c in self.coords:
            s += "{: 13.8f} ".format(c)
        s += " {: 2d}  ".format(-1 if self.flag else 0)
        s += "{}".format(self.name)
        return s

    def __lt__(self, other):
        """
        sorts by canonical smiles invariant, then by name
            more connections first
            then, more non-H bonds first
            then, higher atomic number first
            then, higher number of attached hydrogens first
            then, lower sorting name first
        """
        a = self.get_invariant()
        b = other.get_invariant()
        if a != b:
            return a > b

        a = self.name.split('.')
        b = other.name.split('.')
        while len(a) < len(b):
            a += ['0']
        while len(b) < len(a):
            a += ['0']
        for i, j in zip(a, b):
            if int(i) == int(j):
                continue
            return int(i) < int(j)
        else:
            return True

    def _set_radii(self):
        """Sets atomic radii"""
        try:
            self._radii = float(RADII[self.element])
        except KeyError:
            raise NotImplementedError(
                "Radii not found for element:", self.element)
        return

    def _set_connectivity(self):
        """Sets theoretical maximum connectivity.
        If # connections > self._connectivity, then atom is hyper-valent
        """
        try:
            self._connectivity = int(CONNECTIVITY[self.element])
        except KeyError:
            warn(
                "Connectivity not found for element: " + self.element)
        return

    def add_tag(self, *args):
        for a in args:
            if hasattr(a, '__iter__') and not isinstance(a, str):
                self.tags = self.tags.union(set(a))
            else:
                self.tags.add(a)
        return

    def get_invariant(self):
        """gets initial invariant
        (1) number of connections (d) - nconn
        (2) number of non-hydrogen bonds (dd) - nB
        (3) atomic number (dd) - z
        #(4) sign of charge (d)
        #(5) absolute charge (d)
        (6) number of attached hydrogens (d) - nH
        """
        heavy = [x for x in self.connected if x.element != 'H']
        # number of connected heavy atoms
        nconn = len(heavy)
        # number of connected hydrogens
        nH = len([x for x in self.connected if x.element == 'H'])
        # atomic number
        z = ELEMENTS.index(self.element)
        # number of bonds with heavy atoms
        nB = 0
        for h in heavy:
            nB += BondOrder.get(h, self)

        return "{:1d}{:03d}{:02d}{:1d}".format(int(nconn), int(nB*10), int(z), int(nH))

    # measurement
    def is_connected(self, other, tolerance=0.2):
        """determines if distance between atoms is small enough to be bonded"""
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
                warn_LJ.add(''.join(sorted([self.element, other.element])))
                return 0
        return rv

    def eij(self, other):
        try:
            rv = EIJ[self.element + other.element]
        except KeyError:
            try:
                rv = EIJ[other.element + self.element]
            except KeyError:
                warn_LJ.add(''.join(sorted([self.element, other.element])))
                return 0
        return rv

    def bond_order(self, other):
        return BondOrder.get(self, other)
