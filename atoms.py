"""For parsing and storing atom information"""
import json
import os
import re

import numpy as np

from AaronTools import addlogger
from AaronTools.const import (
    CONNECTIVITY,
    EIJ,
    ELEMENTS,
    MASS,
    RADII,
    RIJ,
    SATURATION,
    TMETAL,
    VDW_RADII,
    COLORS,
)

warn_LJ = set([])

copying_times = {}


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
    def get(cls, a1, a2, dist=None):
        """determines bond order between two atoms based on bond length"""
        try:
            bonds = cls.bonds[cls.key(a1, a2)]
        except KeyError:
            if a1.element == "H" or a2.element == "H":
                return 1
            else:
                BondOrder.warn_atoms.add((a1.element, a2.element))
                return 1
        if dist is None:
            dist = a1.dist(a2)
        closest = 0, None  # (bond order, length diff)
        for order, length in bonds.items():
            diff = abs(length - dist)
            if closest[1] is None or diff < closest[1]:
                closest = order, diff
        return float(closest[0])


@addlogger
class Atom:
    """
    Attributes:

    * element         str
    * coords          np.array(float)
    * flag            bool            true if frozen, false if relaxed
    * name            str             form of \d+(\.\d+)*
    * tags            set
    * charge          float
    * connected       set(Atom)
    * constraint      set(Atom)       for determining constrained bonds
    * _rank
    * _radii          float           for calculating if bonded
    * _connectivity   int             max connections without hypervalence
    * _saturation     int             max connections without hypervalence or charges

    """

    LOG = None

    _bo = BondOrder()

    _do_not_copy = {
        "_hashed",
        "constraint",
        "connected",
        "element",
        "_connectivity",
        "_saturation",
        "_vdw",
        "_radii",
        "_mass",
        "element",
        "coords",
        "name",
        "tags",
        "charge",
        "mass",
    }

    def __init__(
        self, element="", coords=None, flag=False, name="", tags=None, charge=None, mass=None
    ):
        """
        :param str element: element symbol
        :param np.ndarray coords: position
        :param bool flag: whether atom is frozen
        :param str name: atom name
        :param list tags: misc. data
        :param float charge: partial charge of atom
        :param float mass: mass of atom
        """
        
        super().__setattr__("_hashed", False)
        if coords is None:
            coords = []
        # for BqO to have a ghost atom with oxygen basis functions
        ele = str(element).strip()
        element = ele.capitalize()
        if "-" in ele:
            element = "-".join(e.capitalize() for e in ele.split("-"))
        if element.isdigit():
            element = ELEMENTS[int(element)]
        
        if element != "" and element not in ELEMENTS and not element.endswith("Bq"):
            raise ValueError("Unknown element detected: %s" % element)

        if tags is None:
            tags = set()
        elif hasattr(tags, "__iter__") and not isinstance(tags, str):
            tags = set(tags)
        else:
            tags = set([tags])

        if charge is not None:
            charge = float(charge)
        
        self.__dict__.update({
            "element": element,
            "coords": np.array(coords, dtype=float),
            "_mass": mass,
            "name": str(name).strip(),
            "flag": bool(flag),
            "tags": tags,
            "charge": charge,
            "connected": set(),
            "constraint": set(),
            "_rank": None,
        })
        if self.element:
            self.reset()

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
        sorts by canonical smiles rank, then by neighbor ID, then by name
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

        if self._rank is None or other._rank is None:
            # if the ranks are the same, we have little reason to
            # believe the invariants will differ
            a = self.get_invariant()
            b = other.get_invariant()
            if a != b:
                return a > b

        # print("using names")
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

    def __str__(self):
        s = ""
        s += "{:>4s} ".format(self.name)
        s += "{:<3s} ".format(self.element)
        for c in self.coords:
            s += " {: 10.6f}".format(c)
        return s

    def __repr__(self):
        s = ""
        s += "{:>3s} ".format(self.element)
        for c in self.coords:
            s += " {: 13.8f}".format(c)
        s += "  {: 2d}".format(-1 if self.flag else 0)
        s += " {:>4s}".format(self.name)
        s += " ({:d})".format(self._rank) if self._rank is not None else ""
        return s

    @property
    def _radii(self):
        """Sets atomic radii"""
        if self.is_ghost or self.is_dummy:
            return 0
        try:
            return float(RADII[self.element])
        except KeyError:
            self.LOG.warning("Radii not found for element: %s" % self.element)
            return 1.5

    def __setattr__(self, attr, val):
        if self._hashed and attr in {"element", "coords"}:
            raise RuntimeError(
                "Atom %s's Geometry has been hashed and can no longer be changed\n"
                % self.name
                + "setattr was called to set %s to %s" % (attr, val)
            )
        super().__setattr__(attr, val)

    @property
    def _vdw(self):
        """Sets atomic radii"""
        if self.is_ghost or self.is_dummy:
            return 0
        try:
            return float(VDW_RADII[self.element])
        except KeyError:
            self.LOG.warning(
                "VDW Radii not found for element: %s" % self.element
            )
            return 0

    @property
    def _connectivity(self):
        if self.is_ghost or self.is_dummy:
            return 1000
        try:
            return int(CONNECTIVITY[self.element])
        except KeyError:
            return None

    @property
    def _saturation(self):
        """
        Sets theoretical maximum connectivity without the atom having a formal charge.
        
        If # connections > self._saturation, then atom is hyper-valent or has a non-zero formal charge
        """
        if self.is_ghost or self.is_dummy:
            return 0
        try:
            return int(SATURATION[self.element])
        except KeyError:
            return

    @property
    def is_dummy(self):
        return re.match("X$", self.element) is not None

    @property
    def is_ghost(self):
        return re.match("([A-Z][a-z]?-Bq|Bq)", self.element) is not None

    def reset(self):
        pass

    def add_tag(self, *args):
        for a in args:
            if hasattr(a, "__iter__") and not isinstance(a, str):
                self.tags = self.tags.union(set(a))
            else:
                self.tags.add(a)
        return

    def get_invariant(self):
        """
        gets initial invariant, which is formulated using:
        
        #. number of non-hydrogen connections (\d{1}): nconn
        #. sum of bond order of non-hydrogen bonds * 10 (\d{2}): nB
        #. atomic number (\d{3}): z
        #. sign of charge (\d{1}) (not used)
        #. absolute charge (\d{1}) (not used)
        #. number of attached hydrogens (\d{1}): nH
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

    def get_neighbor_id(self):
        """
        gets initial invariant based on self's element and the element of
        the atoms connected to self
        """
        # atomic number
        z = ELEMENTS.index(self.element)
        heavy = [
            ELEMENTS.index(x.element)
            for x in self.connected
            if x.element != "H"
        ]
        # number of non-hydrogen connections
        # number of bonds with heavy atoms and their element
        t = []
        for h in sorted(set(heavy)):
            t.extend([h, heavy.count(h)])

        # number of connected hydrogens
        nH = len(self.connected) - len(heavy)

        fmt = "%03i%02i" + (len(set(heavy)) * "%03i%02i") + "%02i"
        s = fmt % (z, len(heavy), *t, nH)

        return s

    def copy(self):
        """
        creates and returns a copy of self
        """
        rv = Atom(element=self.element, coords=self.coords.copy(), name=self.name, tags=self.tags.copy(), charge=self.charge, mass=self.mass)
        # from time import perf_counter
        for key, val in self.__dict__.items():
            if key in Atom._do_not_copy:
                continue
            # copying_times.setdefault(key, 0)
            # start = perf_counter()
            try:
                rv.__dict__[key] = val.copy()
                # stop = perf_counter()
                # copying_times[key] += (stop - start)
            except AttributeError:
                rv.__dict__[key] = val
                # stop = perf_counter()
                # copying_times[key] += (stop - start)
                # ignore chimerax objects so seqcrow doesn't print a
                # warning when a geometry is copied
                mod = val.__class__.__module__
                if "chimerax" in mod:
                    continue
                if mod != "builtins":
                    self.LOG.warning(
                        "No copy method for {}: in-place changes may occur".format(
                            type(val)
                        )
                    )
        return rv

    # measurement
    def is_connected(self, other, tolerance=None):
        """
        determines if distance between atoms is small enough to be bonded
        same as dist_is_connected but automatically calculates the distance between the atoms
        
        :param float tolerance: buffer for consideration of what is "small enough"; default is 0.3. Cutoff for what constitutes small enough is the sum of the atoms' radii and the tolerance value
        :returns: True if distance is small enough to be bonded, False otherwise
        :rtype: boolean
        """
        return self.dist_is_connected(other, self.dist(other), tolerance)

    def dist_is_connected(self, other, dist_to_other, tolerance):
        """
        determines if distance between atoms is small enough to be bonded. Used to optimize connected checks when distances can be quickly precalculated,
        like with scipy.spatial.distance_matrix

        :param Atom other: atom to measure the distance between
        :param float tolerance: buffer for consideration of what is "small enough"; default is 0.3. Cutoff for what constitutes small enough is the sum of the atoms' radii and the tolerance value
        :param float dist_to_other: distance between the atoms in Angstroms
        :returns: True if distance is small enough to be bonded, False otherwise
        :rtype: boolean
        """
        if tolerance is None:
            tolerance = 0.3

        rad1 = self._radii
        rad2 = other._radii
        if rad1 is None:
            rad1 = 1.5
        if rad2 is None:
            rad2 = 1.5
        cutoff = rad1 + rad2 + tolerance
        return dist_to_other < cutoff

    def add_bond_to(self, other):
        """add self and other to eachother's connected attribute"""
        self.connected.add(other)
        other.connected.add(self)

    def bond(self, other):
        """
        retrieves bond vectors

        :returns: the vector self-->other
        :rtype: np.array
        """
        if isinstance(other, np.ndarray):
            return other - np.array(self.coords)
        return np.array(other.coords) - np.array(self.coords)

    def dist(self, other):
        """
        retrieves distance between Atoms
        
        :returns: the distance between self and other
        :rtype: float
        """
        return np.linalg.norm(self.bond(other))

    def angle(self, a1, a3):
        """
        determines angle between three Atoms

        :returns: the a1-self-a3 angle
        :rtype: float
        """
        v1 = self.bond(a1)
        v2 = self.bond(a3)
        dot = np.dot(v1, v2)
        # numpy is still unhappy with this sometimes
        # every now and again, the changeElement cls test will "fail" b/c
        # numpy throws a warning here
        if abs(dot / (self.dist(a1) * self.dist(a3))) >= 1:
            return 0
        else:
            return np.arccos(dot / (self.dist(a1) * self.dist(a3)))

    @property
    def mass(self):
        """
        retrieves mass of an Atom

        :returns: atomic mass
        :rtype: float
        """
        if self._mass is not None:
            return self._mass
        if self.element in MASS:
            return MASS[self.element]
        elif not (self.is_dummy or self.is_ghost):
            self.LOG.warning("no mass for %s" % self.element)
        return 0

    @mass.setter
    def mass(self, value):
        """
        manually sets the mass of an atom; when retrieving mass values, there is a default value based on the element of the Atom, so this is only necessary if you want an instance of Atom of a differing mass to its typical value.

        :param float value: mass in AMU to set the Atom to;
        """
        self._mass = value

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
        """
        returns dummy atoms in an idealized vsepr geometry
        
        shape_name can be:
        
        * point
        * linear 1
        * linear 2
        * bent 2 tetrahedral
        * bent 2 planar
        * trigonal planar
        * bent 3 tetrahedral
        * t shaped
        * tetrahedral
        * sawhorse
        * seesaw
        * square planar
        * trigonal pyramidal
        * trigonal bipyramidal
        * square pyramidal
        * pentagonal
        * hexagonal
        * trigonal prismatic
        * pentagonal pyramidal
        * octahedral
        * capped octahedral
        * hexagonal pyramidal
        * pentagonal bipyramidal
        * capped trigonal prismatic
        * heptagonal
        * hexagonal bipyramidal
        * heptagonal pyramidal
        * octagonal
        * square antiprismatic
        * trigonal dodecahedral
        * capped cube
        * biaugmented trigonal prismatic
        * cubic
        * elongated trigonal bipyramidal
        * capped square antiprismatic
        * enneagonal
        * heptagonal bipyramidal
        * hula-hoop
        * triangular cupola
        * tridiminished icosahedral
        * muffin
        * octagonal pyramidal
        * tricapped trigonal prismatic
        
        """
        if shape_name == "point":
            return cls.linear_shape()[0:1]
        elif shape_name == "linear 1":
            return cls.linear_shape()[0:2]
        elif shape_name == "linear 2":
            return cls.linear_shape()
        elif shape_name == "bent 2 tetrahedral":
            return cls.tetrahedral_shape()[0:3]
        elif shape_name == "bent 2 planar":
            return cls.trigonal_planar_shape()[0:3]
        elif shape_name == "trigonal planar":
            return cls.trigonal_planar_shape()
        elif shape_name == "bent 3 tetrahedral":
            return cls.tetrahedral_shape()[0:4]
        elif shape_name == "t shaped":
            return cls.octahedral_shape()[0:4]
        elif shape_name == "tetrahedral":
            return cls.tetrahedral_shape()
        elif shape_name == "sawhorse":
            return np.concatenate((
                cls.trigonal_bipyramidal_shape()[0:3,],
                cls.trigonal_bipyramidal_shape()[-2:,],
            ))
        elif shape_name == "seesaw":
            return np.concatenate((
                cls.octahedral_shape()[0:3],
                cls.octahedral_shape()[-2:],
            ))
        elif shape_name == "square planar":
            return cls.octahedral_shape()[0:5]
        elif shape_name == "trigonal pyramidal":
            return cls.trigonal_bipyramidal_shape()[0:5]
        elif shape_name == "trigonal bipyramidal":
            return cls.trigonal_bipyramidal_shape()
        elif shape_name == "square pyramidal":
            return cls.octahedral_shape()[0:6]
        elif shape_name == "pentagonal":
            return cls.pentagonal_bipyramidal_shape()[0:6]
        elif shape_name == "hexagonal":
            return cls.hexagonal_bipyramidal_shape()[0:7]
        elif shape_name == "trigonal prismatic":
            return cls.trigonal_prismatic_shape()
        elif shape_name == "pentagonal pyramidal":
            return cls.pentagonal_bipyramidal_shape()[0:7]
        elif shape_name == "octahedral":
            return cls.octahedral_shape()
        elif shape_name == "capped octahedral":
            return cls.capped_octahedral_shape()
        elif shape_name == "hexagonal pyramidal":
            return cls.hexagonal_bipyramidal_shape()[0:8]
        elif shape_name == "pentagonal bipyramidal":
            return cls.pentagonal_bipyramidal_shape()
        elif shape_name == "capped trigonal prismatic":
            return cls.capped_trigonal_prismatic_shape()
        elif shape_name == "heptagonal":
            return cls.heptagonal_bipyramidal_shape()[0:8]
        elif shape_name == "hexagonal bipyramidal":
            return cls.hexagonal_bipyramidal_shape()
        elif shape_name == "heptagonal pyramidal":
            return cls.heptagonal_bipyramidal_shape()[0:9]
        elif shape_name == "octagonal":
            return cls.octagonal_pyramidal_shape()[0:9]
        elif shape_name == "square antiprismatic":
            return cls.square_antiprismatic_shape()
        elif shape_name == "trigonal dodecahedral":
            return cls.trigonal_dodecahedral_shape()
        elif shape_name == "capped cube":
            return cls.capped_cube_shape()
        elif shape_name == "biaugmented trigonal prismatic":
            return cls.biaugmented_trigonal_prismatic_shape()
        elif shape_name == "cubic":
            return cls.cubic_shape()
        elif shape_name == "elongated trigonal bipyramidal":
            return cls.elongated_trigonal_bipyramidal_shape()
        elif shape_name == "capped square antiprismatic":
            return cls.capped_square_antiprismatic_shape()
        elif shape_name == "enneagonal":
            return cls.enneagonal_shape()
        elif shape_name == "heptagonal bipyramidal":
            return cls.heptagonal_bipyramidal_shape()
        elif shape_name == "hula-hoop":
            return cls.hula_hoop_shape()
        elif shape_name == "triangular cupola":
            return cls.triangular_cupola_shape()
        elif shape_name == "tridiminished icosahedral":
            return cls.tridiminished_icosahedral_shape()
        elif shape_name == "muffin":
            return cls.muffin_shape()
        elif shape_name == "octagonal pyramidal":
            return cls.octagonal_pyramidal_shape()
        elif shape_name == "tricapped trigonal prismatic":
            return cls.tricapped_trigonal_prismatic_shape()
        else:
            raise RuntimeError(
                "no shape method is defined for %s" % shape_name
            )

    @classmethod
    def linear_shape(cls):
        """returns a list of 3 dummy atoms in a linear shape"""
        center = np.zeros(3)
        pos1 = np.array([1.0, 0.0, 0.0])
        pos2 = np.array([-1.0, 0.0, 0.0])

        return np.array([center, pos1, pos2])

    @classmethod
    def trigonal_planar_shape(cls):
        """returns a list of 4 dummy atoms in a trigonal planar shape"""
        positions = cls.trigonal_bipyramidal_shape()
        return positions[:-2,]

    @classmethod
    def tetrahedral_shape(cls):
        """returns a list of 5 dummy atoms in a tetrahedral shape"""
        center = np.zeros(3)
        pos1 = np.array([0.57735, -0.81650, 0.0])
        pos2 = np.array([0.57735, 0.81650, 0.0])
        pos3 = np.array([-0.57735, 0.0, 0.81650])
        pos4 = np.array([-0.57735, 0.0, -0.81650])

        return np.array([center, pos1, pos2, pos3, pos4])

    @classmethod
    def trigonal_bipyramidal_shape(cls):
        """returns a list of 6 dummy atoms in a trigonal bipyramidal shape"""
        center = np.zeros(3)
        pos1 = np.array([1.0, 0.0, 0.0])
        pos2 = np.array([-0.5, 0.86603, 0.0])
        pos3 = np.array([-0.5, -0.86603, 0.0])
        pos4 = np.array([0.0, 0.0, 1.0])
        pos5 = np.array([0.0, 0.0, -1.0])

        return np.array([center, pos1, pos2, pos3, pos4, pos5])

    @classmethod
    def octahedral_shape(cls):
        """returns a list of 7 dummy atoms in an octahedral shape"""
        center = np.zeros(3)
        pos1 = np.array([1.0, 0.0, 0.0])
        pos2 = np.array([0.0, 1.0, 0.0])
        pos3 = np.array([-1.0, 0.0, 0.0])
        pos4 = np.array([0.0, -1.0, 0.0])
        pos5 = np.array([0.0, 0.0, 1.0])
        pos6 = np.array([0.0, 0.0, -1.0])

        return np.array([center, pos1, pos2, pos3, pos4, pos5, pos6])

    @classmethod
    def trigonal_prismatic_shape(cls):
        """returns a list of 7 dummy atoms in a trigonal prismatic shape"""
        center = np.zeros(3)
        pos1 = np.array([-0.6547, -0.3780, 0.6547])
        pos2 = np.array([-0.6547, -0.3780, -0.6547])
        pos3 = np.array([0.6547, -0.3780, 0.6547])
        pos4 = np.array([0.6547, -0.3780, -0.6547])
        pos5 = np.array([0.0, 0.7559, 0.6547])
        pos6 = np.array([0.0, 0.7559, -0.6547])

        return np.array([center, pos1, pos2, pos3, pos4, pos5, pos6])

    @classmethod
    def capped_octahedral_shape(cls):
        """returns a list of 8 dummy atoms in a capped ocrahedral shape"""
        center = np.zeros(3)
        pos1 = np.array([0.0, 0.0, 1.0])
        pos2 = np.array([0.9777, 0.0, 0.2101])
        pos3 = np.array([0.1698, 0.9628, 0.2101])
        pos4 = np.array([-0.9187, 0.3344, 0.2102])
        pos5 = np.array([-0.4888, -0.8467, 0.2102])
        pos6 = np.array([0.3628, -0.6284, -0.6881])
        pos7 = np.array([-0.2601, 0.4505, -0.8540])

        return np.array(
            [center, pos1, pos2, pos3, pos4, pos5, pos6, pos7]
        )

    @classmethod
    def capped_trigonal_prismatic_shape(cls):
        """returns a list of 8 dummy atoms in a capped trigonal prismatic shape"""
        center = np.zeros(3)
        pos1 = np.array([0.0, 0.0, 1.0])
        pos2 = np.array([0.6869, 0.6869, 0.2374])
        pos3 = np.array([-0.6869, 0.6869, 0.2374])
        pos4 = np.array([0.6869, -0.6869, 0.2374])
        pos5 = np.array([-0.6869, -0.6869, 0.2374])
        pos6 = np.array([0.6175, 0.0, -0.7866])
        pos7 = np.array([-0.6175, 0.0, -0.7866])

        return np.array(
            [center, pos1, pos2, pos3, pos4, pos5, pos6, pos7]
        )

    @classmethod
    def pentagonal_bipyramidal_shape(cls):
        """returns a list of 8 dummy atoms in a pentagonal bipyramidal shape"""
        center = np.zeros(3)
        pos1 = np.array([1.0, 0.0, 0.0])
        pos2 = np.array([0.3090, 0.9511, 0.0])
        pos3 = np.array([-0.8090, 0.5878, 0.0])
        pos4 = np.array([-0.8090, -0.5878, 0.0])
        pos5 = np.array([0.3090, -0.9511, 0.0])
        pos6 = np.array([0.0, 0.0, 1.0])
        pos7 = np.array([0.0, 0.0, -1.0])

        return np.array(
            [center, pos1, pos2, pos3, pos4, pos5, pos6, pos7]
        )

    @classmethod
    def biaugmented_trigonal_prismatic_shape(cls):
        """returns a list of 9 dummy atoms in a biaugmented trigonal prismatic shape"""
        center = np.zeros(3)
        pos1 = np.array([-0.6547, -0.3780, 0.6547])
        pos2 = np.array([-0.6547, -0.3780, -0.6547])
        pos3 = np.array([0.6547, -0.3780, 0.6547])
        pos4 = np.array([0.6547, -0.3780, -0.6547])
        pos5 = np.array([0.0, 0.7559, 0.6547])
        pos6 = np.array([0.0, 0.7559, -0.6547])
        pos7 = np.array([0.0, -1.0, 0.0])
        pos8 = np.array([-0.8660, 0.5, 0.0])

        return np.array(
            [center, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8]
        )

    @classmethod
    def cubic_shape(cls):
        """returns a list of 9 dummy atoms in a cubic shape"""
        center = np.zeros(3)
        pos1 = np.array([0.5775, 0.5774, 0.5774])
        pos2 = np.array([0.5775, 0.5774, -0.5774])
        pos3 = np.array([0.5775, -0.5774, 0.5774])
        pos4 = np.array([-0.5775, 0.5774, 0.5774])
        pos5 = np.array([0.5775, -0.5774, -0.5774])
        pos6 = np.array([-0.5775, 0.5774, -0.5774])
        pos7 = np.array([-0.5775, -0.5774, 0.5774])
        pos8 = np.array([-0.5775, -0.5774, -0.5774])

        return np.array(
            [center, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8]
        )

    @classmethod
    def elongated_trigonal_bipyramidal_shape(cls):
        """returns a list of 9 dummy atoms in an elongated trigonal bipyramidal shape"""
        center = np.zeros(3)
        pos1 = np.array([0.6547, 0.0, 0.7559])
        pos2 = np.array([-0.6547, 0.0, 0.7559])
        pos3 = np.array([0.6547, 0.6547, -0.3780])
        pos4 = np.array([-0.6547, 0.6547, -0.3780])
        pos5 = np.array([0.6547, -0.6547, -0.3780])
        pos6 = np.array([-0.6547, -0.6547, -0.3780])
        pos7 = np.array([1.0, 0.0, 0.0])
        pos8 = np.array([-1.0, 0.0, 0.0])

        return np.array(
            [center, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8]
        )

    @classmethod
    def hexagonal_bipyramidal_shape(cls):
        """returns a list of 9 dummy atoms in a hexagonal bipyramidal shape"""
        center = np.zeros(3)
        pos1 = np.array([0.0, -1.0, 0.0])
        pos2 = np.array([0.8660, -0.5, 0.0])
        pos3 = np.array([0.8660, 0.5, 0.0])
        pos4 = np.array([0.0, 1.0, 0.0])
        pos5 = np.array([-0.8660, 0.5, 0.0])
        pos6 = np.array([-0.8660, -0.5, 0.0])
        pos7 = np.array([0.0, 0.0, 1.0])
        pos8 = np.array([0.0, 0.0, -1.0])

        return np.array(
            [center, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8]
        )

    @classmethod
    def square_antiprismatic_shape(cls):
        """returns a list of 9 dummy atoms in a square antiprismatic shape"""
        center = np.zeros(3)
        pos1 = np.array([0.0, 0.0, 1.0])
        pos2 = np.array([0.9653, 0.0, 0.2612])
        pos3 = np.array([-0.5655, 0.7823, 0.2612])
        pos4 = np.array([-0.8825, -0.3912, 0.2612])
        pos5 = np.array([0.1999, -0.9444, 0.2612])
        pos6 = np.array([0.3998, 0.7827, -0.4776])
        pos7 = np.array([-0.5998, 0.1620, -0.7836])
        pos8 = np.array([0.4826, -0.3912, -0.7836])

        return np.array(
            [center, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8]
        )

    @classmethod
    def trigonal_dodecahedral_shape(cls):
        """returns a list of 9 dummy atoms in a trigonal dodecahedral shape"""
        center = np.zeros(3)
        pos1 = np.array([-0.5997, 0.0, 0.8002])
        pos2 = np.array([0.0, -0.9364, 0.3509])
        pos3 = np.array([0.5998, 0.0, 0.8002])
        pos4 = np.array([0.0, 0.9364, 0.3509])
        pos5 = np.array([-0.9364, 0.0, -0.3509])
        pos6 = np.array([0.0, -0.5997, -0.8002])
        pos7 = np.array([0.9365, 0.0, -0.3509])
        pos8 = np.array([0.0, 0.5997, -0.8002])

        return np.array(
            [center, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8]
        )

    @classmethod
    def heptagonal_bipyramidal_shape(cls):
        """returns a list of 10 dummy atoms in a heptagonal bipyramidal shape"""
        center = np.zeros(3)
        pos1 = np.array([1.0, 0.0, 0.0])
        pos2 = np.array([0.6235, 0.7818, 0.0])
        pos3 = np.array([-0.2225, 0.9749, 0.0])
        pos4 = np.array([-0.9010, 0.4339, 0.0])
        pos5 = np.array([-0.9010, -0.4339, 0.0])
        pos6 = np.array([-0.2225, -0.9749, 0.0])
        pos7 = np.array([0.6235, -0.7818, 0.0])
        pos8 = np.array([0.0, 0.0, 1.0])
        pos9 = np.array([0.0, 0.0, -1.0])

        return np.array(
            [center, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9]
        )

    @classmethod
    def capped_cube_shape(cls):
        """returns a list of 10 dummy atoms in a capped cube shape"""
        center = np.zeros(3)
        pos1 = np.array([0.6418, 0.6418, 0.4196])
        pos2 = np.array([0.6418, -0.6418, 0.4196])
        pos3 = np.array([-0.6418, 0.6418, 0.4196])
        pos4 = np.array([-0.6418, -0.6418, 0.4196])
        pos5 = np.array([0.5387, 0.5387, -0.6478])
        pos6 = np.array([0.5387, -0.5387, -0.6478])
        pos7 = np.array([-0.5387, 0.5387, -0.6478])
        pos8 = np.array([-0.5387, -0.5387, -0.6478])
        pos9 = np.array([0.0, 0.0, 1.0])

        return np.array(
            [center, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9]
        )

    @classmethod
    def capped_square_antiprismatic_shape(cls):
        """returns a list of 10 dummy atoms in a capped square antiprismatic shape"""
        center = np.zeros(3)
        pos1 = np.array([0.9322, 0.0, 0.3619])
        pos2 = np.array([-0.9322, 0.0, 0.3619])
        pos3 = np.array([0.0, 0.9322, 0.3619])
        pos4 = np.array([0.0, -0.9322, 0.3619])
        pos5 = np.array([0.5606, 0.5606, -0.6095])
        pos6 = np.array([-0.5606, 0.5606, -0.6095])
        pos7 = np.array([-0.5606, -0.5606, -0.6095])
        pos8 = np.array([0.5606, -0.5606, -0.6095])
        pos9 = np.array([0.0, 0.0, 1.0])

        return np.array(
            [center, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9]
        )

    @classmethod
    def enneagonal_shape(cls):
        """returns a list of 10 dummy atoms in an enneagonal shape"""
        center = np.zeros(3)
        pos1 = np.array([1.0, 0.0, 0.0])
        pos2 = np.array([0.7660, 0.6428, 0.0])
        pos3 = np.array([0.1736, 0.9848, 0.0])
        pos4 = np.array([-0.5, 0.8660, 0.0])
        pos5 = np.array([-0.9397, 0.3420, 0.0])
        pos6 = np.array([-0.9397, -0.3420, 0.0])
        pos7 = np.array([-0.5, -0.8660, 0.0])
        pos8 = np.array([0.1736, -0.9848, 0.0])
        pos9 = np.array([0.7660, -0.6428, 0.0])

        return np.array(
            [center, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9]
        )

    @classmethod
    def hula_hoop_shape(cls):
        """returns a list of 10 dummy atoms in a hula hoop shape"""
        center = np.zeros(3)
        pos1 = np.array([1.0, 0.0, 0.0])
        pos2 = np.array([0.5, 0.8660, 0.0])
        pos3 = np.array([-0.5, 0.8660, 0.0])
        pos4 = np.array([-1.0, 0.0, 0.0])
        pos5 = np.array([-0.5, -0.8660, 0.0])
        pos6 = np.array([0.5, -0.8660, 0.0])
        pos7 = np.array([0.0, 0.0, 1.0])
        pos8 = np.array([0.5, 0.0, -0.8660])
        pos9 = np.array([-0.5, 0.0, -0.8660])

        return np.array(
            [center, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9]
        )

    @classmethod
    def triangular_cupola_shape(cls):
        """ returns a list of 10 dummy atoms in a triangular cupola shape"""
        center = np.zeros(3)
        pos1 = np.array([1.0, 0.0, 0.0])
        pos2 = np.array([0.5, 0.8660, 0.0])
        pos3 = np.array([-0.5, 0.8660, 0.0])
        pos4 = np.array([-1.0, 0.0, 0.0])
        pos5 = np.array([0.5, -0.8660, 0.0])
        pos6 = np.array([-0.5, -0.8660, 0.0])
        pos7 = np.array([0.5, 0.2887, -0.8165])
        pos8 = np.array([-0.5, 0.2887, -0.8165])
        pos9 = np.array([0.0, -0.5774, -0.8165])

        return np.array(
            [center, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9]
        )

    @classmethod
    def tridiminished_icosahedral_shape(cls):
        """returns a list of 10 dummy atoms in a tridiminished icosahedral shape"""
        center = np.zeros(3)
        pos1 = np.array([-0.2764, 0.8507, -0.4472])
        pos2 = np.array([-0.8944, 0.0, -0.4472])
        pos3 = np.array([-0.2764, -0.8507, -0.4472])
        pos4 = np.array([0.7236, -0.5257, -0.4472])
        pos5 = np.array([0.8944, 0.0, 0.4472])
        pos6 = np.array([0.2764, 0.8507, 0.4472])
        pos7 = np.array([-0.7236, -0.5257, 0.4472])
        pos8 = np.array([0.0, 0.0, 1.0])
        pos9 = np.array([0.0, 0.0, -1.0])

        return np.array(
            [center, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9]
        )

    @classmethod
    def muffin_shape(cls):
        """returns a list of 10 dummy atoms in a muffin shape"""
        center = np.zeros(3)
        pos1 = np.array([0.0, 0.9875, 0.1579])
        pos2 = np.array([0.9391, 0.3051, 0.1579])
        pos3 = np.array([0.5804, -0.7988, 0.1579])
        pos4 = np.array([-0.5804, -0.7988, 0.1579])
        pos5 = np.array([-0.9391, 0.3055, 0.1579])
        pos6 = np.array([-0.5799, -0.3356, -0.7423])
        pos7 = np.array([0.5799, -0.3356, -0.7423])
        pos8 = np.array([0.0, 0.6694, -0.7429])
        pos9 = np.array([0.0, 0.0, 1.0])

        return np.array(
            [center, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9]
        )

    @classmethod
    def octagonal_pyramidal_shape(cls):
        """returns a list of 10 dummy atoms in an octagonal pyramidal shape"""
        center = np.zeros(3)
        pos1 = np.array([0.7071, -0.7071, 0.0])
        pos2 = np.array([1.0, 0.0, 0.0])
        pos3 = np.array([0.7071, 0.7071, 0.0])
        pos4 = np.array([0.0, 1.0, 0.0])
        pos5 = np.array([-0.7071, 0.7071, 0.0])
        pos6 = np.array([-1.0, 0.0, 0.0])
        pos7 = np.array([-0.7071, -0.7071, 0.0])
        pos8 = np.array([0.0, -1.0, 0.0])
        pos9 = np.array([0.0, 0.0, -1.0])

        return np.array(
            [center, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9]
        )

    @classmethod
    def tricapped_trigonal_prismatic_shape(cls):
        """returns a list of 10 dummy atoms in a tricapped trigonal prismatic shape"""
        center = np.zeros(3)
        pos1 = np.array([0.0, 0.0, 1.0])
        pos2 = np.array([-0.2357, 0.9129, 0.3333])
        pos3 = np.array([-0.9428, 0.0, 0.3333])
        pos4 = np.array([0.2357, -0.9129, 0.3333])
        pos5 = np.array([0.9428, 0.0, 0.3333])
        pos6 = np.array([0.5303, 0.6847, -0.5])
        pos7 = np.array([-0.5303, -0.6847, -0.5])
        pos8 = np.array([-0.5893, 0.4564, -0.6667])
        pos9 = np.array([0.5893, -0.4564, -0.6667])

        return np.array(
            [center, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9]
        )

    @staticmethod
    def new_shape(old_shape, new_connectivity, bond_change):
        """
        returns the name of the expected vsepr geometry when the number of bonds
        changes by +/- 1

        :param str old_shape: vsepr geometry name
        :param int new_connectivity: connectivity (see Atom._connectivity)
        :param int bond_change: +1 or -1, indicating that the number of bonds is changing by 1
        
        """
        if old_shape == "point":
            if bond_change == 1:
                return "linear 1"
            else:
                return None

        elif old_shape == "linear 1":
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

        elif old_shape == "bent 2 tetrahedral":
            if bond_change == 1:
                return "bent 3 tetrahedral"
            elif bond_change == -1:
                return "linear 1"

        elif old_shape == "bent 2 planar":
            if bond_change == 1:
                return "trigonal planar"
            elif bond_change == -1:
                return "linear 1"

        elif old_shape == "trigonal planar":
            if bond_change == 1:
                return "tetrahedral"
            elif bond_change == -1:
                if new_connectivity == 4:
                    return "bent 2 tetrahedral"
                else:
                    return "bent 2 planar"

        elif old_shape == "bent 3 tetrahedral":
            if bond_change == 1:
                return "tetrahedral"
            elif bond_change == -1:
                return "bent 2 tetrahedral"

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
                return "trigonal bipyramidal"
            elif bond_change == -1:
                return "bent 3 tetrahedral"

        elif old_shape == "square planar":
            if bond_change == 1:
                return "trigonal bipyramidal"
            elif bond_change == -1:
                return "t shaped"

        elif old_shape == "trigonal bipyramidal":
            if bond_change == 1:
                return "octahedral"
            elif bond_change == -1:
                return "sawhorse"

        elif old_shape == "octahedral":
            if bond_change == -1:
                return "trigonal bipyramid"

        else:
            raise RuntimeError("no shape method is defined for %s" % old_shape)

    def get_vsepr(self):
        """
        determine vsepr geometry around an atom
        
        :returns: 
            * :shape, score: as a string and the score assigned to that shape
            * :None: if self has > 6 bonds
        
        scores > 0.5 are generally questionable
        
        see atom.get_shape for a list of shapes
        """

        # shapes with a code in the commend next to them are from Simas et. al. Inorg. Chem. 2018, 57, 17, 10557–10567

        # determine what geometries to try based on the number of bonded atoms
        try_shapes = {}
        if len(self.connected) == 0:
            try_shapes["point"] = Atom.get_shape("point")

        elif len(self.connected) == 1:
            try_shapes["linear 1"] = Atom.get_shape("linear 1")

        elif len(self.connected) == 2:
            try_shapes["linear 2"] = Atom.get_shape("linear 2")
            try_shapes["bent 2 planar"] = Atom.get_shape("bent 2 planar")
            try_shapes["bent 2 tetrahedral"] = Atom.get_shape(
                "bent 2 tetrahedral"
            )

        elif len(self.connected) == 3:
            try_shapes["trigonal planar"] = Atom.get_shape("trigonal planar")
            try_shapes["bent 3 tetrahedral"] = Atom.get_shape(
                "bent 3 tetrahedral"
            )
            try_shapes["t shaped"] = Atom.get_shape("t shaped")

        elif len(self.connected) == 4:
            try_shapes["tetrahedral"] = Atom.get_shape("tetrahedral")
            try_shapes["sawhorse"] = Atom.get_shape("sawhorse")
            try_shapes["seesaw"] = Atom.get_shape("seesaw")
            try_shapes["square planar"] = Atom.get_shape("square planar")
            try_shapes["trigonal pyramidal"] = Atom.get_shape(
                "trigonal pyramidal"
            )

        elif len(self.connected) == 5:
            try_shapes["trigonal bipyramidal"] = Atom.get_shape(
                "trigonal bipyramidal"
            )
            try_shapes["square pyramidal"] = Atom.get_shape("square pyramidal")
            try_shapes["pentagonal"] = Atom.get_shape("pentagonal")  # PP-5

        elif len(self.connected) == 6:
            try_shapes["octahedral"] = Atom.get_shape("octahedral")
            try_shapes["hexagonal"] = Atom.get_shape("hexagonal")  # HP-6
            try_shapes["trigonal prismatic"] = Atom.get_shape(
                "trigonal prismatic"
            )  # TPR-6
            try_shapes["pentagonal pyramidal"] = Atom.get_shape(
                "pentagonal pyramidal"
            )  # PPY-6
        elif len(self.connected) == 7:
            try_shapes["capped octahedral"] = Atom.get_shape(
                "capped octahedral"
            )  # COC-7
            try_shapes["capped trigonal prismatic"] = Atom.get_shape(
                "capped trigonal prismatic"
            )  # CTPR-7
            try_shapes["heptagonal"] = Atom.get_shape("heptagonal")  # HP-7
            try_shapes["hexagonal pyramidal"] = Atom.get_shape(
                "hexagonal pyramidal"
            )  # HPY-7
            try_shapes["pentagonal bipyramidal"] = Atom.get_shape(
                "pentagonal bipyramidal"
            )  # PBPY-7
        elif len(self.connected) == 8:
            try_shapes["biaugmented trigonal prismatic"] = Atom.get_shape(
                "biaugmented trigonal prismatic"
            )  # BTPR-8
            try_shapes["cubic"] = Atom.get_shape("cubic")  # CU-8
            try_shapes["elongated trigonal bipyramidal"] = Atom.get_shape(
                "elongated trigonal bipyramidal"
            )  # ETBPY-8
            try_shapes["hexagonal bipyramidal"] = Atom.get_shape(
                "hexagonal bipyramidal"
            )  # HBPY-8
            try_shapes["heptagonal pyramidal"] = Atom.get_shape(
                "heptagonal pyramidal"
            )  # HPY-8
            try_shapes["octagonal"] = Atom.get_shape("octagonal")  # OP-8
            try_shapes["square antiprismatic"] = Atom.get_shape(
                "square antiprismatic"
            )  # SAPR-8
            try_shapes["trigonal dodecahedral"] = Atom.get_shape(
                "trigonal dodecahedral"
            )  # TDD-8
        elif len(self.connected) == 9:
            try_shapes["capped cube"] = Atom.get_shape("capped cube")  # CCU-9
            try_shapes["capped square antiprismatic"] = Atom.get_shape(
                "capped square antiprismatic"
            )  # CSAPR-9
            try_shapes["enneagonal"] = Atom.get_shape("enneagonal")  # EP-9
            try_shapes["heptagonal bipyramidal"] = Atom.get_shape(
                "heptagonal bipyramidal"
            )  # HBPY-9
            try_shapes["hula-hoop"] = Atom.get_shape("hula-hoop")  # HH-9
            try_shapes["triangular cupola"] = Atom.get_shape(
                "triangular cupola"
            )  # JTC-9
            try_shapes["tridiminished icosahedral"] = Atom.get_shape(
                "tridiminished icosahedral"
            )  # JTDIC-9
            try_shapes["muffin"] = Atom.get_shape("muffin")  # MFF-9
            try_shapes["octagonal pyramidal"] = Atom.get_shape(
                "octagonal pyramidal"
            )  # OPY-9
            try_shapes["tricapped trigonal prismatic"] = Atom.get_shape(
                "tricapped trigonal prismatic"
            )  # TCTPR-9

        else:
            return None, None

        # make a copy of the atom and the atoms bonded to it
        # set each bond length to 1 to more easily compare to the
        # idealized shapes from Atom
        adjusted_shape = [atom.copy() for atom in [self, *self.connected]]
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
            X = try_shapes[shape]
            r2 = np.matmul(np.transpose(X), X)
            u, s2, vh = np.linalg.svd(r2)

            score = sum(np.abs(s1 - s2))
            if best_score is None or score < best_score:
                best_score = score
                best_shape = shape

        return best_shape, best_score


    def draw_atom(self, ax, fp = 40, ascale = 0.5, linewidth=0.1):
        """
        Draw HoukMol style atom and add to Matplotlib axis ax

        :param matplotlib.pyplot.Axis ax: Matplotlib axis object
        :param float fp: z-value for focal point to add simple perspective (Default: 40)
        :param float ascale: scaling factor for covalent radii (Default: 0.5)
        :param float linewidth: linewidth (in points) for atom details (default: 0.1)
        """

        try:
            import matplotlib.patches as patches
        except:
            self.LOG.error("Must install matplot lib")
            return None

        x, y, z = self.coords
        r = RADII[self.element]*ascale*(fp + z)/fp
        color=COLORS[self.element]

        # draw circle and two ellipses for atom plus two circles for specular highlight
        circle = patches.Circle((x, y), r, facecolor=color, edgecolor='black', lw=linewidth, zorder=z)
        acc = patches.Circle((x+r/6, y+r/4), 0.2*r, alpha=0.7, facecolor='white', edgecolor="None", zorder=z)
        acc2 = patches.Circle((x+r/6, y+r/4), 0.3*r, alpha=0.3, facecolor='white', edgecolor="None", zorder=z)
        arc1 = patches.Arc((x, y), r, 2*r, theta1=270, theta2=90, edgecolor='black', lw=linewidth, zorder=z)
        arc2 = patches.Arc((x, y), 2*r, r, theta1=180, theta2=360, edgecolor='black', lw=linewidth, zorder=z)

        # add atom componets to axis
        ax.add_patch(circle)
        ax.add_patch(acc)
        ax.add_patch(acc2)
        ax.add_patch(arc1)
        ax.add_patch(arc2)


