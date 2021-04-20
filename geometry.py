"""For storing, manipulating, and measuring molecular structures"""
import itertools
import re
from collections import deque
from copy import copy as shallow_copy
from copy import deepcopy
from warnings import warn
from urllib.request import urlopen
from urllib.error import HTTPError

import numpy as np

import AaronTools
import AaronTools.utils.utils as utils
from AaronTools.atoms import Atom
from AaronTools.const import ELEMENTS
from AaronTools.fileIO import FileReader, FileWriter
from AaronTools.utils.prime_numbers import Primes
from AaronTools.oniomatoms import OniomAtom

LAH_bonded_to = re.compile("(LAH) bonded to ([0-9]+)")
LA_on = re.compile("(LA) on ([0-9])")
COORD_THRESHOLD = 0.2


class Geometry:
    """
    Attributes:
        name
        comment
        atoms
        other
    """

    Primes()

    def __init__(
        self, structure="", name="", comment="", refresh_connected=True
    ):
        """
        :structure: can be a Geometry(), a FileReader(), a file name, or a
            list of atoms
        """
        self.name = name
        self.comment = comment
        self.atoms = []

        if isinstance(structure, Geometry):
            # new from geometry
            self.atoms = structure.atoms
            if not name:
                self.name = structure.name
            if not comment:
                self.comment = structure.comment
            return
        elif isinstance(structure, FileReader):
            # get info from FileReader object
            from_file = structure
        elif isinstance(structure, str) and structure != "":
            # parse file
            from_file = FileReader(structure)
        elif hasattr(structure, "__iter__") and structure != "":
            for a in structure:
                if not isinstance(a, Atom):
                    raise TypeError
            else:
                # list of atoms supplied
                self.atoms = structure
                if refresh_connected:
                    self.refresh_connected()
                return
        else:
            return


        # only get here if we were given a file reader object or a file name
        self.name = from_file.name
        self.comment = from_file.comment
        self.atoms = from_file.atoms
        if refresh_connected:
            self.other = self.parse_comment()
            self.refresh_connected()
        return

    # attribute access
    def _stack_coords(self, atoms=None):
        """
        Generates a N x 3 coordinate matrix for atoms
        Note: the matrix rows are copies of, not references to, the
            Atom.coords objects. Run Geometry.update_geometry(matrix) after
            using this method to save changes.
        """
        if atoms is None:
            atoms = self.atoms
        else:
            atoms = self.find(atoms)
        rv = np.zeros((len(atoms), 3), dtype=float)
        for i, a in enumerate(atoms):
            rv[i] = a.coords[:]
        return rv

    def elements(self):
        """ returns list of elements composing the atoms in the geometry """
        return [a.element for a in self.atoms]

    def coords(self, atoms=None):
        """
        returns N x 3 coordinate matrix for requested atoms
            (defaults to all atoms)
        """
        if atoms is None:
            return self._stack_coords(self.atoms)
        return self._stack_coords(atoms)

    # utilities
    def __eq__(self, other):
        """
        two geometries equal if:
            same number of atoms
            coordinates of atoms similar (now distance, soon RMSD)
        """
        if id(self) == id(other):
            return True
        if len(self.atoms) != len(other.atoms):
            return False
        rmsd = self.RMSD(other, name_sort=True)
        return rmsd < COORD_THRESHOLD

    def __add__(self, other):
        if isinstance(other, Atom):
            other = [other]
        elif not isinstance(other, list):
            other = other.atoms
        self.atoms += other
        return self

    def __sub__(self, other):
        if isinstance(other, Atom):
            other = [other]
        elif not isinstance(other, list):
            other = other.atoms
        for o in other:
            self.atoms.remove(o)
        for a in self.atoms:
            if a.connected & set(other):
                a.connected = a.connected - set(other)
        return self

    def __repr__(self):
        """ string representation """
        s = ""
        for a in self.atoms:
            s += a.__repr__() + "\n"
        return s

    def write(self, name=None, style="xyz", *args, **kwargs):
        """
        write geometry to a file
        parameters:
            name (str): defaults to self.name
            style (str): defaults to xyz
        See fileIO.FileWriter for more details
        """
        tmp = self.name
        if name is not None:
            self.name = name
        out = FileWriter.write_file(self, style, *args, **kwargs)
        self.name = tmp

        if out is not None:
            return out

    def copy(self, atoms=None, name=None, comment=None):
        """
        creates a new copy of the geometry
        parameters:
            atoms (list): defaults to all atoms
            name (str): defaults to NAME_copy
        """
        if name is None:
            name = self.name
        if comment is None:
            comment = self.comment
        atoms = self._fix_connectivity(atoms)

        return Geometry(atoms, name, comment, refresh_connected=False)

    def _fix_connectivity(self, atoms=None, copy=True):
        """
        for fixing the connectivity for a set of atoms when grabbing
        a fragment or copying atoms, ensures atom references are sane

        :atoms: the atoms to fix connectivity for; connections to atoms
            outside of this list are severed in the resulting list
        :copy: perform a deepcopy of the atom list
        """
        if atoms is None:
            atoms = self.atoms
        else:
            atoms = self.find(atoms)

        connectivity = []
        for a in atoms:
            connectivity += [
                [atoms.index(i) for i in a.connected if i in atoms]
            ]
        if copy:
            atoms = deepcopy(atoms)
        for a, con in zip(atoms, connectivity):
            a.connected = set([])
            for c in con:
                a.connected.add(atoms[c])
        return atoms

    def update_geometry(self, structure):
        """
        Replace current coords with those from :structure:

        :structure: a file name, atom list, Geometry or np.array() of shape Nx3
        """
        if isinstance(structure, np.ndarray):
            coords = structure
            elements = None
        else:
            tmp = Geometry(structure)
            elements = [a.element for a in tmp.atoms]
            coords = tmp._stack_coords()
        if coords.shape[0] != len(self.atoms):
            raise RuntimeError(
                "Updated geometry has different number of atoms"
            )
        for i, row in enumerate(coords):
            if elements is not None and elements[i] != self.atoms[i].element:
                raise RuntimeError(
                    "Updated coords atom order doesn't seem to match original "
                    + "atom order. Stopping..."
                )
            self.atoms[i].coords = row
        self.refresh_connected()
        return

    def find(self, *args, debug=False):
        """
        finds atom in geometry
        Parameters:
            *args are tags, names or elements
            args=(['this', 'that'], 'other') will find atoms for which
                ('this' || 'that') && 'other' == True
        Returns:
            [Atom()] or []
        Raises:
            LookupError when the tags/names provided do not exist
            However, it will return empty list if valid tag/names were provided
                but were screened out using the && argument form
        """

        def _find(arg):
            """ find a single atom """
            if isinstance(arg, Atom):
                return [arg]
            rv = []
            name_str = re.compile("^(\*|\d)+(\.?\*|\.\d+)*$")
            if isinstance(arg, str) and name_str.match(arg) is not None:
                test_name = arg.replace(".", "\.")
                test_name = test_name.replace("*", "(\.?\d+\.?)*")
                test_name = re.compile("^" + test_name + "$")
                # this is a name
                for a in self.atoms:
                    if test_name.search(a.name) is not None:
                        rv += [a]

            elif isinstance(arg, str) and len(arg.split(",")) > 1:
                list_style = arg.split(",")
                if len(list_style) > 1:
                    for i in list_style:
                        if len(i.split("-")) > 1:
                            rv += _find_between(i)
                        else:
                            rv += _find(i)

            elif isinstance(arg, str) and len(arg.split('-')) > 1:
                rv += _find_between(arg)

            elif isinstance(arg, str) and arg in ELEMENTS:
                # this is an element
                for a in self.atoms:
                    if a.element == arg:
                        rv += [a]
            else:
                # this is a tag
                for a in self.atoms:
                    if arg in a.tags:
                        rv += [a]
            return rv

        def _find_between(arg):
            """ find sequence of atoms """

            def _name2ints(name):
                name = name.split(".")
                return [int(i) for i in name]

            a1, a2 = tuple(arg.split("-"))
            a1 = _find(a1)[0]
            a2 = _find(a2)[0]

            rv = []
            for a in self.atoms:
                # keep if a.name is between a1.name and a2.name
                test_name = _name2ints(a.name)
                a1_name = _name2ints(a1.name)
                a2_name = _name2ints(a2.name)

                for tn, a1n, a2n in zip(test_name, a1_name, a2_name):
                    if tn < a1n:
                        # don't want if test atom sorts before a1
                        break
                    if tn > a2n:
                        # don't want if test atom sorts after a2
                        break
                else:
                    rv += _find(a)
            return rv

        if len(args) == 1:
            if isinstance(args[0], tuple):
                args = args[0]
        rv = []
        for a in args:
            if hasattr(a, "__iter__") and not isinstance(a, str):
                # argument is a list of sub-arguments
                # OR condition
                tmp = []
                for i in a:
                    tmp += _find(i)
                rv += [tmp]
            else:
                rv += [_find(a)]

        # error if no atoms found (no error if AND filters out all found atoms)
        if len(rv) == 1:
            if len(rv[0]) == 0:
                raise LookupError("Could not find atom: %s on\n%s\n%s" % (str(args), self.name, str(self)))
            return rv[0]

        # exclude atoms not fulfilling AND requirement
        tmp = []
        for i in rv[0]:
            good = True
            for j in rv[1:]:
                if i not in j:
                    good = False
            if good:
                tmp += [i]
        return tmp

    def find_exact(self, *args):
        """
        finds exactly the same number of atoms as arguments used.
        Raises LookupError if wrong number of atoms found
        """
        rv = []
        err = "Wrong number of atoms found: "
        is_err = False
        for arg in args:
            try:
                a = self.find(arg)
            except LookupError:
                a = []

            if len(a) != 1:
                is_err = True
                err += "{} (found {}), ".format(arg, len(a))
            else:
                rv += a

        if is_err:
            err = err[:-2]
            raise LookupError(err)
        return tuple(rv)

    def refresh_connected(self, threshold=None, rank=True):
        """
        reset connected atoms
        atoms are connected if their distance from each other is less than
            the sum of their covalent radii plus a threshold
        """
        # clear current connectivity
        old_connectivity = []
        for a in self.atoms:
            old_connectivity += [a.connected]
            a.connected = set([])

        # determine connectivity
        refresh_ranks = False
        for i, a in enumerate(self.atoms):
            for b in self.atoms[i + 1:]:
                if a.is_connected(b, threshold):
                    a.connected.add(b)
                    b.connected.add(a)
            if not refresh_ranks and a.connected ^ old_connectivity[i]:
                refresh_ranks = True

        # get ranks
        if refresh_ranks and rank:
            self.refresh_ranks()

    def refresh_ranks(self):
        rank = self.canonical_rank()
        for a, r in zip(self.atoms, rank):
            a._rank = r
        return

    def canonical_rank(self, heavy_only=False):
        """
        put atoms in canonical smiles rank
        (follows algorithm described in 10.1021/ci00062a008)
        """
        primes = Primes.list(len(self.atoms))
        atoms = {}  # {atom: rank}
        ranks = {}  # {rank: [atom]}

        def get_rank(atoms):
            new_atoms = {}
            new_ranks = {}
            rank_key = sorted(set(atoms.values()))

            # looping through self.atoms should ensure that random flips
            # between two tied atoms doesn't happen?
            for a in self.atoms:
                if heavy_only and a.element == "H":
                    continue
                val = rank_key.index(atoms[a])
                new_atoms[a] = val
                if val in new_ranks:
                    new_ranks[val] += [a]
                else:
                    new_ranks[val] = [a]

            return new_atoms, new_ranks

        # use invariants as initial rank
        for a in self.atoms:
            if heavy_only and a.element == "H":
                continue
            atoms[a] = a.get_invariant()

        atoms, ranks = get_rank(atoms)

        # use neighbors to break ties
        count = 0
        while count < 50:
            count += 1
            new_atoms = {}
            for a in atoms:
                # new rank is product of neighbors' prime rank
                val = primes[atoms[a]]
                for c in a.connected:
                    if heavy_only and c.element == "H":
                        continue
                    val *= primes[atoms[c]]
                new_atoms[a] = val
            atoms, new_ranks = get_rank(new_atoms)
            if new_ranks == ranks:
                break
            if sorted(new_ranks.keys()) == sorted(ranks.keys()):
                for a in new_atoms:
                    new_atoms[a] *= 2
                new_atoms[ranks[0][0]] -= 1
            atoms, new_ranks = get_rank(new_atoms)
            ranks = new_ranks
        else:
            warn_str = "\nMax number of canonical ranking cycles exceeded: {}"
            warn_str = warn_str.format(self.name)
            warn(warn_str)

        rv = []
        for a in self.atoms:
            if a in atoms:
                rv += [atoms[a]]
        return rv

    def reorder(
        self, start=None, targets=None, canonical=True, heavy_only=False
    ):
        """
        Depth-first reorder of atoms based on canonical ranking
        if canonical is True (default):
            starts at lowest canonical rank (use when invariance desired)
        if canonical is False:
            starts at highest canonical rank (use when more central atoms
            should come first)
        """

        def rank_sort(targets, reverse=False):
            try:
                return sorted(targets, key=lambda a: a._rank, reverse=reverse)
            except TypeError:
                return sorted(targets, reverse=not reverse)

        def find_min(targets):
            return rank_sort(targets)[0]

        def find_max(targets):
            return rank_sort(targets)[-1]

        if targets is None:
            targets = self.atoms
        else:
            targets = self.find(targets)
        if heavy_only:
            targets = [t for t in targets if t.element != "H"]

        non_targets = [a for a in self.atoms if a not in targets]

        # get starting atom
        if start is None:
            if canonical:
                order = [find_min(targets)]
            else:
                order = [find_max(targets)]
        else:
            order = self.find(start)
        start = rank_sort(order, reverse=canonical)[0]
        stack = rank_sort(start.connected, reverse=canonical)
        atoms_left = set(targets) - set(order) - set(stack)
        while len(stack) > 0:
            this = stack.pop()
            if heavy_only and this.element == "H":
                continue
            order += [this]
            connected = this.connected & atoms_left
            atoms_left -= set(connected)
            stack += rank_sort(connected, reverse=canonical)

            if len(stack) == 0 and len(atoms_left) > 0:
                stack += [find_min(atoms_left)]
                atoms_left -= set(stack)

        return order, non_targets

    def get_constraints(self):
        rv = set([])
        for i, a in enumerate(self.atoms[:-1]):
            if not a.constraint:
                continue
            for j, b in enumerate(self.atoms[i:]):
                if b not in a.constraint:
                    continue
                rv.add((i, i + j))
        return sorted(rv)

    def get_connectivity(self):
        rv = []
        for atom in self.atoms:
            rv += [atom.connected]
        return rv

    def parse_comment(self):
        """
        Saves auxillary data found in comment line
        """
        if not self.comment:
            return {}
        rv = {}
        # constraints
        match = re.search("F:([0-9;-]+)", self.comment)
        if match is not None:
            rv["constraint"] = []
            for a in self.atoms:
                a.constraint = set([])
            match = match.group(1).split(";")
            for m in match:
                if m == "":
                    continue
                m = m.split("-")
                m = [int(i) - 1 for i in m]
                for i, j in zip(m[:-1], m[1:]):
                    a = self.atoms[i]
                    b = self.atoms[j]
                    a.constraint.add(b)
                    b.constraint.add(a)
                rv["constraint"] += [m]
        # active centers
        match = re.search("C:([0-9,]+)", self.comment)
        if match is not None:
            rv["center"] = []
            match = match.group(1).split(",")
            for m in match:
                if m == "":
                    continue
                a = self.atoms[int(m) - 1]
                a.add_tag("center")
                rv["center"] += [a]

        # ligand
        match = re.search("L:([0-9-,]+)", self.comment)
        if match is not None:
            rv["ligand"] = []
            match = match.group(1).split(",")
            for m in match:
                if m == "":
                    continue
                m = m.split("-")
                for i in range(int(m[0]) - 1, int(m[1])):
                    try:
                        a = self.atoms[i]
                    except IndexError:
                        continue
                    a.add_tag("ligand")
                    rv["ligand"] += [a]

        #link atoms
        match = re.search("LA:([0-9;-]+)", self.comment)
        if match is not None:
            rv["link_atoms"] = []
            match = match.group(1).split(";")
            for m in match:
                if m == "":
                    continue
                m = m.split("-")
                m = [int(i) - 1 for i in m]
                rv["link_atoms"] += [m]
                for i, j in zip(m[:-1], m[1:]):
                    a = self.atoms[i]
                    b = self.atoms[j]
                    a.add_tag("LAH bonded to " + b.name)
                    b.add_tag("bonded to LA on " + a.name)

        # key atoms
        match = re.search("K:([0-9,;]+)", self.comment)
        if match is not None:
            rv["key_atoms"] = []
            match = match.group(1).split(";")
            for m in match:
                if m == "":
                    continue
                m = m.split(",")
                for i in m:
                    if i.strip() == "":
                        continue
                    rv["key_atoms"] += [int(i) - 1]
        self.other = rv
        return rv

    def _flag(self, flag, targets=None):
        """
        freezes targets if <flag> is True,
        relaxes targets if <flag> is False
        """
        if targets is not None:
            targets = self.find(targets)
        else:
            targets = self.atoms

        for a in targets:
            a.flag = flag
        return

    def freeze(self, targets=None):
        """
        freezes atoms in the geometry
        """
        self._flag(True, targets)

    def relax(self, targets=None):
        """
        relaxes atoms in the geometry
        """
        self._flag(False, targets)

    def LJ_energy(self, other=None):
        """
        computes LJ energy using autodock parameters
        """

        def calc_LJ(a, b):
            sigma = a.rij(b)
            epsilon = a.eij(b)
            dist = a.dist(b)
            return epsilon * ((sigma / dist) ** 12 - (sigma / dist) ** 6)

        energy = 0
        for i, a in enumerate(self.atoms):
            if other is None:
                try:
                    tmp = self.atoms[i + 1:]
                except IndexError:
                    return energy
            else:
                try:
                    tmp = other.atoms
                except AttributeError:
                    tmp = other

            for b in tmp:
                if a == b:
                    continue
                energy += calc_LJ(a, b)

        return energy

    # geometry measurement
    def bond(self, a1, a2):
        """ takes two atoms and returns the bond vector """
        a1, a2 = self.find_exact(a1, a2)
        return a1.bond(a2)

    def angle(self, a1, a2, a3=None):
        """returns a1-a2-a3 angle"""
        a1, a2, a3 = self.find_exact(a1, a2, a3)
        return a2.angle(a1, a3)

    def dihedral(self, a1, a2, a3, a4):
        """measures dihedral angle of a1 and a4 with respect to a2-a3 bond"""
        a1, a2, a3, a4 = self.find_exact(a1, a2, a3, a4)

        b12 = a1.bond(a2)
        b23 = a2.bond(a3)
        b34 = a3.bond(a4)

        dihedral = np.cross(np.cross(b12, b23), np.cross(b23, b34))
        dihedral = np.dot(dihedral, b23) / np.linalg.norm(b23)
        dihedral = np.arctan2(
            dihedral, np.dot(np.cross(b12, b23), np.cross(b23, b34))
        )

        return dihedral

    def COM(self, targets=None, heavy_only=False, mass_weight=False):
        """
        calculates center of mass of the target atoms
        returns a vector from the origin to the center of mass
        parameters:
            targets (list) - the atoms to use in calculation (defaults to all)
            heavy_only (bool) - exclude hydrogens (defaults to False)
        """
        # get targets
        if targets is not None:
            targets = self.find(targets)
        else:
            targets = self.atoms
        # screen hydrogens if necessary
        if heavy_only:
            targets = [a for a in targets if a.element != "H"]

        # COM = (1/M) * sum(m * r) = sum(m*r) / sum(m)
        total_mass = 0
        center = np.array([0, 0, 0], dtype=np.float)
        for t in targets:
            if mass_weight:
                total_mass += t.mass()
                center += t.mass() * t.coords
            else:
                center += t.coords

        if mass_weight:
            return center / total_mass
        return center / len(targets)

    def RMSD(
        self,
        ref,
        align=False,
        heavy_only=False,
        targets=None,
        ref_targets=None,
        name_sort=False,
        sort=False,
        longsort=False,
    ):
        """
        calculates the RMSD between two geometries
        Returns: rmsd (float)

        :ref: (Geometry) the geometry to compare to
        :align: (bool) if True (default), align self to other;
            if False, just calculate the RMSD
        :heavy_only: (bool) only use heavy atoms (default False)
        :targets: (list) the atoms in `self` to use in calculation
        :ref_targets: (list) the atoms in the reference geometry to use
        :sort: (bool) sort atoms before comparing
        :longsort: (bool) use a more expensive but better sorting method
            (only use for small molecules!)
        """

        def _RMSD(ref, other):
            """
            ref and other are lists of atoms
            returns rmsd, angle, vector
                rmsd (float)
                angle (float) angle to rotate by
                vector (np.array(float)) the rotation axis
            """
            matrix = np.zeros((4, 4), dtype=np.float64)
            for i, a in enumerate(ref):
                pt1 = a.coords
                try:
                    pt2 = other[i].coords
                except IndexError:
                    break
                matrix += utils.quat_matrix(pt2, pt1)

            eigenval, eigenvec = np.linalg.eigh(matrix)
            val = eigenval[0]
            vec = eigenvec.T[0]

            if val > 0:
                # val is the SD
                rmsd = np.sqrt(val / len(ref))
            else:
                # negative numbers that are basically zero, like -1e-16
                rmsd = 0

            # sometimes it freaks out if the coordinates are right on
            # top of each other and gives overly large rmsd/rotation
            # I think this is a numpy precision problem, may want to
            # try scipy.linalg to see if that helps?
            tmp = sum(
                [
                    np.linalg.norm(a.coords - b.coords) ** 2
                    for a, b in zip(ref, other)
                ]
            )
            tmp = np.sqrt(tmp / len(ref))
            if tmp < rmsd:
                rmsd = tmp
                vec = np.array([0, 0, 0])
            return rmsd, vec

        def get_orders(obj, targets):
            """ get orders starting at different atoms """
            # try just regular canonical ordering
            orders = [obj.reorder(targets=targets, canonical=False)[0]]
            if not longsort:
                # and other orderings starting with each start atom
                for t in targets:
                    if t.element == "H":
                        continue
                    tmp = [
                        obj.reorder(start=t, targets=targets, canonical=False)[
                            0
                        ]
                    ]
                return orders
            else:
                # This way might be better, as there could be canonical ties,
                # but it gets really costly really fast. Maybe there's a way to
                # slim it down?

                order = orders[-1]
                swap = []
                tmp = set([])
                for i, o in enumerate(order):
                    if i == 0:
                        last_rank = order[0]._rank
                        continue
                    if o._rank == last_rank:
                        tmp.add(i - 1)
                        tmp.add(i)
                    elif tmp:
                        swap += [tmp.copy()]
                        tmp = set([])
                    last_rank = o._rank
                pairs = []
                for n in range(1, len(swap) + 1):
                    for to_swap in itertools.combinations(swap, n):
                        for s in to_swap:
                            for pair in itertools.combinations(list(s), 2):
                                pairs += [pair]
                for i, j in pairs:
                    tmp = order.copy()
                    tmp[i], tmp[j] = tmp[j], tmp[i]
                    orders += [tmp]
                return orders

        # get target atoms
        if longsort:
            sort = True
        if sort:
            name_sort = True
        tmp = targets
        if targets is not None:
            targets = self.find(targets)
        else:
            targets = self.atoms
        if ref_targets is not None:
            ref_targets = ref.find(ref_targets)
        elif tmp is not None:
            ref_targets = ref.find(tmp)
        else:
            ref_targets = ref.atoms

        # screen out hydrogens if heavy_only requested
        if heavy_only:
            targets = [a for a in targets if a.element != "H"]
            ref_targets = [a for a in ref_targets if a.element != "H"]

        # align center of mass to origin
        com = self.COM(targets=targets)
        ref_com = ref.COM(targets=ref_targets)

        self.coord_shift(-com)
        ref.coord_shift(-ref_com)

        # sort targets if requested and perform rmsd calculation
        if name_sort:
            ref_targets = sorted(ref_targets, key=lambda atom: float(atom))
            targets = sorted(targets, key=lambda atom: float(atom))

        if sort:
            # try current ordering
            min_rmsd = _RMSD(ref_targets, targets)

            # get other orderings
            ref_targets = [
                a for a in ref.reorder(canonical=False)[0] if a in ref_targets
            ]

            ref_orders = get_orders(ref, ref_targets)
            orders = get_orders(self, targets)

            # find min RMSD for each ordering
            for r in ref_orders:
                for o in orders:
                    o = [a for a in o if a in targets]
                    tmp = _RMSD(ref_targets, o)
                    if min_rmsd is None or tmp[0] < min_rmsd[0]:
                        min_rmsd = tmp
            rmsd, vec = min_rmsd
        else:
            rmsd, vec = _RMSD(ref_targets, targets)

        # return rmsd
        ref.coord_shift(ref_com)
        if not align:
            self.coord_shift(com)
            return rmsd
        # or update geometry and return rmsd
        if np.linalg.norm(vec) > 0:
            self.rotate(vec)
        self.coord_shift(ref_com)
        return rmsd

    # geometry manipulation
    def get_fragment(self, start, stop, as_object=False, copy=False):
        """
        Returns:
            [Atoms()] if as_object == False
            Geometry() if as_object == True

        :start: the atoms to start on
        :stop: the atom(s) to avoid
        :as_object: return as list (default) or Geometry object
        :copy: whether or not to copy the atoms before returning the list;
            copy will automatically fix connectivity information
        """
        # TODO: allow stop=None, default to smallest fragment,
        # with override option
        start = self.find(start)
        stop = self.find(stop)

        stack = deque(start)
        frag = start
        while len(stack) > 0:
            connected = stack.popleft()
            connected = connected.connected - set(stop) - set(frag)
            stack.extend(sorted(connected))
            frag += sorted(connected)

        if as_object:
            return self.copy(atoms=frag, comment="")
        if copy:
            return self._fix_connectivity(frag, copy=True)
        return frag

    def remove_fragment(self, start, avoid, add_H=True):
        """
        Removes a fragment of the geometry
        Returns:
            (list) :start: + the removed fragment

        :start: the atom of the fragment to be removed that attaches to the
            rest of the geometry
        :avoid: the atoms :start: is attached to that should be avoided
        :add_H: default is to change :start: to H and update bond lengths, but
            add_H=False overrides this behaviour

        """
        start = self.find(start)
        avoid = self.find(avoid)
        frag = self.get_fragment(start, avoid)[len(start):]
        self -= frag

        # replace start with H
        rv = start + frag
        for a in start:
            if not add_H:
                break
            a.element = "H"
            a._set_radii()
            self.change_distance(a, a.connected - set(frag), fix=2)
        return rv

    def coord_shift(self, vector, targets=None):
        """
        shifts the coordinates of the target atoms by a vector
        parameters:
            vector (np.array) - the shift vector
            targets (list) - the target atoms to shift (default to all)
        """
        if targets is None:
            targets = self.atoms
        else:
            targets = self.find(targets)

        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float)
        for t in targets:
            t.coords += vector
        return

    def change_distance(
        self, a1, a2, dist=None, adjust=False, fix=0, as_group=True
    ):
        """For setting/adjusting bond length between atoms
        Parameters:
            a1_arg - the first atom
            a2_arg - the second atom
            dist - the distance to change by/to. Default is to set the bond
                   length to that determined by RADII
            adjust - default is to set the bond length to `dist`, adjust=True
                     indicates the current bond length should be adjusted by
                     `dist`
            fix - default is to move both a1 and a2 by half of `dist`, fix=1
                  will move only a2 and fix=2 will move only a1
            as_group - default is to move the fragments connected to a1 and a2
                       as well, as_group=False will only move the requested
                       atom(s)
        """
        a1, a2 = self.find_exact(a1, a2)

        # determine new bond length
        if dist is None:
            new_dist = a1._radii + a2._radii
        elif adjust:
            new_dist = a1.dist(a2) + dist
        else:
            new_dist = dist
        dist = a1.dist(a2)

        # adjustment vector for each atom
        adj_a1 = (new_dist - dist) * a2.bond(a1) / a2.dist(a1)
        adj_a2 = (new_dist - dist) * a1.bond(a2) / a1.dist(a2)
        if fix == 0:
            adj_a1 /= 2
            adj_a2 /= 2
        elif fix == 1:
            adj_a1 = None
        elif fix == 2:
            adj_a2 = None
        else:
            raise ValueError(
                "Bad parameter `fix` (should be 0, 1, or 2):", fix
            )

        # get atoms to adjust
        if as_group:
            a1 = self.get_fragment(a1, a2)
            a2 = self.get_fragment(a2, a1)
        else:
            a1 = [a1]
            a2 = [a2]

        # translate atom(s)
        for i in a1:
            if adj_a1 is None:
                break
            i.coords += adj_a1
        for i in a2:
            if adj_a2 is None:
                break
            i.coords += adj_a2

        return

    def rotate(self, w, angle=None, targets=None, center=None):
        """
        rotates target atoms by an angle about an axis

        :w: (np.array) - the axis of rotation (doesnt need to be unit vector)
            or a quaternion (angle not required then)
        :angle: (float) - the angle by which to rotate (in radians)
        :targets: (list) - the atoms to rotate (defaults to all)
        :center: (Atom or list) - if provided, the atom (or COM of a list)
            will be centered at the origin before rotation, then shifted
            back after rotation
        """
        if targets is None:
            targets = self.atoms
        else:
            targets = self.find(targets)

        # shift geometry to place center atom at origin
        if center is not None:
            if not isinstance(center, type(self.atoms[0].coords)):
                tmp = self.find(center)
                if len(tmp) > 1:
                    center = deepcopy(self.COM(tmp))
                else:
                    center = deepcopy(tmp[0].coords)
            else:
                center = deepcopy(center)
            self.coord_shift(-1 * center)

        if not isinstance(w, np.ndarray):
            w = np.array(w, dtype=np.double)

        if angle is not None and len(w) == 3:
            w /= np.linalg.norm(w)
            q = np.hstack(([np.cos(angle / 2)], w * np.sin(angle / 2)))
        elif len(w) != 4:
            raise TypeError(
                """Vector `w` must be either a rotation vector (len 3)
                or a quaternion (len 4). Angle parameter required if `w` is a
                rotation vector"""
            )
        else:
            q = w

        q /= np.linalg.norm(q)
        qs = q[0]
        qv = q[1:]

        for t in targets:
            t.coords = (
                t.coords
                + 2 * qs * np.cross(qv, t.coords)
                + 2 * np.cross(qv, np.cross(qv, t.coords))
            )

        if center is not None:
            self.coord_shift(center)

    def change_angle(
        self,
        a1,
        a2,
        a3,
        angle,
        radians=True,
        adjust=False,
        fix=0,
        as_group=True,
    ):
        """For setting/adjusting angle between atoms
        Parameters:
            a1 - the first atom
            a2 - the second atom (vertex)
            a3 - the third atom
            angle - the angle to change by/to
            radians - default units are radians, radians=False uses degrees
            adjust - default is to set the angle to `angle`, adjust=True
                indicates the current angle should be adjusted by `angle`
            fix - default is to move both a1 and a3 by half of `angle`, fix=1
                will move only a3 and fix=3 will move only a1
            as_group - default is to move the fragments connected to a1 and a3
                as well, as_group=False will only move the requested atom(s)
        """
        try:
            a1, a2, a3 = self.find([a1, a2, a3])
        except ValueError:
            raise LookupError(
                "Bad atom request: {}, {}, {}".format(a1, a2, a3)
            )

        # get rotation vector
        v1 = a2.bond(a1)
        v2 = a2.bond(a3)
        w = np.cross(v1, v2)
        w = w / np.linalg.norm(w)

        # determine rotation angle
        if not radians:
            angle = np.deg2rad(angle)
        if not adjust:
            angle -= a2.angle(a1, a3)

        # get target fragments
        a1_frag = self.get_fragment(a1, a2)
        a3_frag = self.get_fragment(a3, a2)

        # shift a2 to origin
        self.coord_shift(-a2.coords, a1_frag)
        self.coord_shift(-a2.coords, a3_frag)

        # perform rotation
        if fix == 0:
            angle /= 2
            self.rotate(w, -angle, a1_frag)
            self.rotate(w, angle, a3_frag)
        elif fix == 1:
            self.rotate(w, angle, a3_frag)
        elif fix == 3:
            self.rotate(w, -angle, a1_frag)
        else:
            raise ValueError("fix must be 0, 1, 3 (supplied: {})".format(fix))

        # shift a2 back to original location
        self.coord_shift(a2.coords, a1_frag)
        self.coord_shift(a2.coords, a3_frag)

    def change_dihedral(
        self,
        a1,
        a2,
        *a3_a4_dihedral,
        radians=True,
        adjust=False,
        fix=0,
        as_group=True
    ):
        """For setting/adjusting dihedrals
        Parameters:
            a1 - the first atom
            a2 - the second atom
            a3 - the third atom (optional for adjust=True if as_group=True)
            a4 - the fourth atom (optional for adjust=True if as_group=True)
            dihedral - the dihedral to change by/to
            radians - default units are radians, radians=False uses degrees
            adjust - default is to set the dihedral to `dihedral`, adjust=True
                indicates the current dihedral should be adjusted by `dihedral`
            fix - default is to move both a1 and a3 by half of `dihedral`,
                fix=1 will move only a3 and fix=3 will move only a1
            as_group - default is to move the fragments connected to a1 and a3
                as well, as_group=False will only move the requested atom(s)
        """
        # get atoms
        count = len(a3_a4_dihedral)
        if count == 1 and adjust and as_group:
            # we can just define the bond to rotate about, as long as we are
            # adjusting, not setting, the whole fragments on either side
            dihedral, = a3_a4_dihedral
            a2, a3 = (a1, a2)
            a2, a3 = self.find_exact(a2, a3)
            a1, a4 = (next(iter(a2.connected)), next(iter(a3.connected)))
        elif count != 3:
            raise TypeError(
                "Number of atom arguments provided insufficient to define "
                + "dihedral"
            )
        else:
            a3, a4, dihedral = a3_a4_dihedral
            a1, a2, a3, a4 = self.find_exact(a1, a2, a3, a4)
        # get fragments
        if as_group:
            a2_frag = self.get_fragment(a2, a3)[1:]
            a3_frag = self.get_fragment(a3, a2)[1:]
        else:
            a2_frag = [a1]
            a3_frag = [a4]

        # fix units
        if not radians:
            dihedral = np.deg2rad(dihedral)
        # get adjustment
        if not adjust:
            dihedral -= self.dihedral(a1, a2, a3, a4)

        # rotate fragments
        self.coord_shift(-a2.coords, a2_frag)
        self.coord_shift(-a3.coords, a3_frag)
        if fix == 0:
            dihedral /= 2
            self.rotate(a2.bond(a3), -dihedral, a2_frag)
            self.rotate(a2.bond(a3), dihedral, a3_frag)
        elif fix == 1:
            self.rotate(a2.bond(a3), dihedral, a3_frag)
        elif fix == 4:
            self.rotate(a2.bond(a3), -dihedral, a2_frag)
        else:
            raise ValueError(
                "`fix` must be 0, 1, or 4 (supplied: {})".format(fix)
            )
        self.coord_shift(a2.coords, a2_frag)
        self.coord_shift(a3.coords, a3_frag)

    def minimize_torsion(self, targets, axis, center, geom=None):
        """
        Rotate :targets` to minimize the LJ potential

        :targets: the target atoms to rotate
        :axis: the axis by which to rotate
        :center: where to center before rotation
        :geom: calculate LJ potential between self and another geometry-like
            object, instead of just within self
        """
        targets = self.find(targets)
        if geom is None:
            geom = self
        increment = 5
        E_min = None
        angle_min = None

        # rotate targets by increment and save lowest energy
        angle = 0
        for inc in range(0, 360, increment):
            angle += increment
            self.rotate(
                axis, np.rad2deg(increment), targets=targets, center=center
            )
            energy = self.LJ_energy(geom)

            if E_min is None or energy < E_min:
                E_min = energy
                angle_min = angle

        # rotate to min angle
        self.rotate(
            axis, np.rad2deg(angle_min - angle), targets=targets, center=center
        )

        return

    def substitute(self, sub, target, attached_to=None):
        """
        substitutes fragment containing `target` with substituent `sub`
        if end provided, this is the atom where the substituent is attached
        if end==None, replace the smallest fragment containing `target`
        """
        # set up substituent object
        sub.refresh_connected()

        # determine target and atoms defining connection bond
        target = self.find(target)

        # attached_to is provided or is the atom giving the
        # smallest target fragment
        if attached_to is not None:
            attached_to = self.find_exact(attached_to)
        else:
            smallest_frag = None
            smallest_attached_to = None
            # get all possible connection points
            attached_to = set()
            for t in target:
                attached_to = attached_to | (t.connected - set(target))
            # find smallest fragment
            for e in attached_to:
                frag = self.get_fragment(target, e)
                if smallest_frag is None or len(frag) < len(smallest_frag):
                    smallest_frag = frag
                    smallest_attached_to = e
            attached_to = [smallest_attached_to]
        if len(attached_to) != 1:
            raise NotImplementedError(
                "Can only replace substituents with one point of attachment"
            )
        attached_to = attached_to[0]
        sub.end = attached_to

        # determine which atom of target fragment is connected to attached_to
        sub_attach = attached_to.connected & set(target)
        if len(sub_attach) > 1:
            raise NotImplementedError(
                "Can only replace substituents with one point of attachment"
            )
        if len(sub_attach) < 1:
            raise LookupError("attached_to atom not connected to targets")
        sub_attach = sub_attach.pop()

        # manipulate substituent geometry; want sub.atoms[0] -> sub_attach
        #   attached_to == sub.end
        #   sub_attach will eventually be sub.atoms[0]
        # move attached_to to the origin
        shift = attached_to.coords.copy()
        self.coord_shift(-1 * shift)
        # align substituent to current bond
        bond = self.bond(attached_to, sub_attach)
        sub.align_to_bond(bond)
        # shift geometry back and shift substituent to appropriate place
        self.coord_shift(shift)
        sub.coord_shift(shift)

        # tag and update name for sub atoms
        for s in sub.atoms:
            s.add_tag(sub.name)
            s.name = sub_attach.name + "." + s.name

        #add first atom of new substituent where the target atom was
        self.atoms.insert(self.atoms.index(target[0]), sub.atoms[0])
        # remove old substituent
        self.remove_fragment(target, attached_to, add_H=False)
        self -= target
        attached_to.connected.discard(sub_attach)

        # fix connections
        attached_to.connected.add(sub.atoms[0])
        sub.atoms[0].connected.add(attached_to)

        # add the rest of the new substituent at the end
        self += sub.atoms[1:]
        # fix bond distance
        self.change_distance(attached_to, sub.atoms[0], as_group=True, fix=1)

    def ring_substitute(self, targets, ring_fragment):
        """take ring, reorient it, put it on self"""

        def attach_short(geom, walk, ring_fragment):
            """for when walk < end, rmsd and remove end[1:-1]"""
            ring_fragment.RMSD(geom, align=True, targets=ring_fragment.end, ref_targets=walk)

            ring_waddle(geom, targets, [walk[1], walk[-2]], ring_fragment)

            for atom in ring_fragment.end[1:-1]:
                for t in atom.connected:
                    if t not in ring_fragment.end:
                        ring_fragment -= t

                ring_fragment -= atom

            ring_fragment.end = walk[1:-1]

            r = geom.remove_fragment([walk[0], walk[-1]], walk[1:-1], add_H=False)
            geom -= [walk[0], walk[-1]]

            geom.atoms.extend(ring_fragment.atoms)
            geom.refresh_connected()

        def ring_waddle(geom, targets, walk_end, ring):
            """adjusted the new bond lengths by moving the ring in a 'waddling' motion"""
            d0 = walk_end[0].dist(targets[0])
            
            if hasattr(targets[0], "_radii") and hasattr(walk_end[0], "_radii"):
                d0_exp = targets[0]._radii + walk_end[0]._radii
            else:
                d0_exp = d0

            if hasattr(ring.end[0], "_radii") and hasattr(walk_end[0], "_radii"):
                d1_exp = ring.end[0]._radii + walk_end[0]._radii
            else:
                d1_exp = ring.end[0].dist(walk_end[0])

            d1 = (d0/d0_exp) * d1_exp

            v1 = ring.end[-1].bond(walk_end[0])
            v2 = ring.end[-1].bond(ring.end[0])

            v1_n = np.linalg.norm(v1)
            v2_n = np.linalg.norm(v2)

            target_angle = np.arccos((d1**2 - v1_n**2 - v2_n**2) / (-2. * v1_n * v2_n))
            current_angle = ring.end[-1].angle(ring.end[0], walk_end[0])
            ra = target_angle - current_angle

            rv = np.cross(v1, v2)

            ring.rotate(rv, ra, center=ring.end[-1])


            d0 = walk_end[-1].dist(targets[-1])
            
            if hasattr(targets[-1], "_radii") and hasattr(walk_end[-1], "_radii"):
                d0_exp = targets[-1]._radii + walk_end[-1]._radii
            else:
                d0_exp = d0

            if hasattr(ring.end[-1], "_radii") and hasattr(walk_end[-1], "_radii"):
                d1_exp = ring.end[-1]._radii + walk_end[-1]._radii
            else:
                d1_exp = ring.end[-1].dist(walk_end[-1])

            d1 = (d0/d0_exp) * d1_exp

            v1 = ring.end[0].bond(walk_end[-1])
            v2 = ring.end[0].bond(ring.end[-1])

            v1_n = np.linalg.norm(v1)
            v2_n = np.linalg.norm(v2)

            target_angle = np.arccos((d1**2 - v1_n**2 - v2_n**2) / (-2. * v1_n * v2_n))
            current_angle = ring.end[0].angle(ring.end[-1], walk_end[-1])
            ra = target_angle - current_angle

            rv = np.cross(v1, v2)

            ring.rotate(rv, ra, center=ring.end[0])

        targets = self.find(targets)

        #find a path between the targets
        walk = self.short_walk(*targets)
        if len(ring_fragment.end) != len(walk):
            ring_fragment.find_end(len(walk), start=ring_fragment.end)

        if len(walk) == len(ring_fragment.end) and len(walk) != 2:
            attach_short(self, walk, ring_fragment)

        elif len(walk[1:-1]) == 0:
            raise ValueError("insufficient information to close ring - selected atoms are bonded to each other: %s" % \
                    (" ".join(str(a) for a in targets)))

        else:
            raise ValueError("this ring is not appropriate to connect\n%s\nand\n%s:\n%s\nspacing is %i; expected %i" % \
                    (targets[0], targets[1], ring_fragment.name, len(ring_fragment.end), len(walk)))

    def short_walk(self, atom1, atom2):
        """try to find the shortest path between atom1 and atom2"""
        a1 = self.find(atom1)[0]
        a2 = self.find(atom2)[0]
        l = [a1]
        start = a1
        max_iter = len(self.atoms)
        i = 0
        while start != a2:
            i += 1
            if i > max_iter:
                raise LookupError("could not determine best path between %s and %s" % (str(atom1), str(atom2)))
            v1 = start.bond(a2)
            max_overlap = None
            for atom in start.connected:
                if atom not in l:
                    v2 = start.bond(atom)
                    overlap = np.dot(v1, v2)
                    if max_overlap is None or overlap > max_overlap:
                        new_start = atom
                        max_overlap = overlap

            l.append(new_start)
            start = new_start

        return l

    @classmethod
    def from_string(cls, name, form='smiles'):
        """get structure from string
        form=iupac -> iupac to smiles from opsin API
                       --> form=smiles
        form=smiles -> structure from cactvs API"""

        accepted_forms = ['iupac', 'smiles']

        if form not in accepted_forms:
            raise NotImplementedError("cannot create substituent given %s; use one of %s" % form, str(accepted_forms))


        if form == 'smiles':
            smiles = name
        elif form == 'iupac':
            #opsin seems to be better at iupac names with radicals
            url_smi = "https://opsin.ch.cam.ac.uk/opsin/%s.smi" % name

            try:
                smiles = urlopen(url_smi).read().decode('utf8')
            except HTTPError:
               raise RuntimeError("%s is not a valid IUPAC name or https://opsin.ch.cam.ac.uk is down" % name)

        url_sd = "https://cactus.nci.nih.gov/chemical/structure/%s/file?format=sdf" % smiles
        s_sd = urlopen(url_sd).read().decode('utf8')
        f = FileReader((name, "sd", s_sd))
        return Geometry(f)

    def oniom_frag(self, layer="", as_object=False):
        frag=[]
        self.sub_links()
        if layer not in ['H', 'L', 'M']:
            raise ValueError("Error in layer request")
        else:
            self.parse_comment()
            for a in self.atoms:
                if isinstance(a, OniomAtom) and a.layer==layer:
                    frag.append(a)
                elif not isinstance (a, OniomAtom):
                    raise TypeError("Atom does not have property 'layer'")
 
        if as_object:
            #need to write function to write a comment based on tags
            frag = Geometry(structure=frag).add_links().sub_hosts()
        return frag

    def add_links(self):
        tmp=[]
        for a in self.atoms:
            if "LAH" in str(a.tags):
                c = OniomAtom(element="H", coords = a.coords, layer="H")
                c.add_tag("LA " + a.name)
                tmp.append(c)
        self = self + tmp
        return self

    def sub_links(self):
        tmp = []
        for a in self.atoms:
            match = re.search("LA [0-9]+", str(a.tags))
            if match is not None:
                tmp.append(a)
            else:
                pass
        self = self - tmp
        return self

    def sub_hosts(self):
        tmp = []
        for a in self.atoms:
            match = re.search("LAH", str(a.tags))
            if match is not None:
                tmp.append(a)
            else:
                pass
        self = self - tmp
        return self

    def check_links(self):
        for i in range(len(self.atoms)):
            if not isinstance(self.atoms[i], OniomAtom):
                raise TypeError("geometry must be composed of OniomAtoms")
            else:
                numtrue = 0
                numlinks = 0
                for j in range(i+1, len(self.atoms)):
                    if (self.atoms[i].is_connected(self.atoms[j])) \
                            and (self.atoms[i].layer != self.atoms[j].layer):
                        a = self.atoms[i]
                        b = self.atoms[b]
                        numlinks += 1
                        if a > b:
                            match = LAH_bonded_to.search(str(b.tags))
                            match2 = LA_on.search(str(a.tags))
                            if match.group(2) == a.name and match2.group(2) == b.name:
                                numtrue += 1
                        elif b > a:
                            match = LAH_bonded_to.search(str(a.tags))
                            match2 = LA_on.search(str(b.tags))
                            if match.group(2) == b.name and match2.group(2) == a.name:
                                numtrue += 1
        return numtrue == numlinks

    #def write_comment(self):
    #    for atom in self.atoms:
    #        if "constraint" in str(atom.tags):
    #            do something coming soon

    def update_names(self):
        for i, a in enumerate(self.atoms):
            if a.name:
                pass
            else:
                a.name = str(i + 1)
        return self

