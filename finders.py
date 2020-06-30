"""finders are used by Geometry.find to locate atoms in a more general way"""
import numpy as np


class Finder:
    def get_matching_atoms(self, atoms, geometry=None):
        """overwrite with function that returns list(Atom) of the atoms that
        match your Finder's criteria
        geometry is an optional argument that could be used to e.g. find
        atoms a certain number of bonds"""
        pass


class BondsFrom(Finder):
    """exact number of bonds from specified atom"""
    def __init__(self, atom, number_of_bonds):
        super().__init__()

        self.central_atom = atom
        self.n_bonds = number_of_bonds

    def __str__(self):
        return "atoms %i bonds of %s" % (self.n_bonds, self.central_atom)

    def get_matching_atoms(self, atoms, geometry):
        """returns List(Atom) that are a certain number of bonds away from the given atom"""
        matching_atoms = []
        for atom in atoms:
            try:
                path = geometry.shortest_path(atom, self.central_atom)
            except LookupError:
                continue

            if len(path) - 1 == self.n_bonds:
                matching_atoms.append(atom)

        return matching_atoms


class WithinBondsOf(BondsFrom):
    """within a specified number of bonds from the atom"""
    def __init__(self, atom, number_of_bonds):
        super().__init__(atom, number_of_bonds)

    def __str__(self):
        return "atoms %i bonds of %s" % (self.n_bonds, self.central_atom)

    def get_matching_atoms(self, atoms, geometry):
        """returns List(Atom) that are a certain number of bonds away from the given atom"""
        matching_atoms = []
        for atom in atoms:
            try:
                path = geometry.shortest_path(atom, self.central_atom)
            except LookupError:
                continue

            if len(path) - 1 <= self.n_bonds:
                matching_atoms.append(atom)

        return matching_atoms


class BondedTo(Finder):
    """returns all atoms that are bonded to the specified atom"""
    def __init__(self, atom):
        super().__init__()

        self.atom = atom

    def __str__(self):
        return "atoms bonded to %s" % self.atom

    def get_matching_atoms(self, atoms, geometry=None):
        """returns list(Atom) that are within a radius of a point"""
        return [atom for atom in atoms if atom in self.atom.connected]


class WithinRadiusFromPoint(Finder):
    """within a specified radius of a point"""
    def __init__(self, point, radius):
        super().__init__()

        self.point = np.array(point)
        self.radius = radius
    
    def __str__(self):
        return "atoms within %.2f angstroms of (%.2f, %.2f, %.2f)" % (self.radius, *self.point)

    def get_matching_atoms(self, atoms, geometry=None):
        """returns list(Atom) that are within a radius of a point"""
        matching_atoms = []
        for atom in atoms:
            coords = atom.coords
            d = np.linalg.norm(coords - self.point)
            if d < self.radius:
                matching_atoms.append(atom)

        return matching_atoms


class WithinRadiusFromAtom(Finder):
    """within a specified radius of a point"""
    def __init__(self, atom, radius):
        super().__init__()

        self.point = atom
        self.radius = radius
    
    def __str__(self):
        return "atoms within %.2f angstroms of %s" % (self.radius, self.atom)

    def get_matching_atoms(self, atoms, geometry=None):
        """returns list(Atom) that are within a radius of a point"""
        matching_atoms = []
        for atom in atoms:
            coords = atom.coords
            d = self.atom.dist(atom)
            if d < self.radius:
                matching_atoms.append(atom)

        return matching_atoms
    

class NotAny(Finder):
    """atoms not matching specifiers/Finders"""
    def __init__(self, *args):
        """args can be any number of Finders and/or other atom specifiers (tags, elements, etc.)"""
        super().__init__()

        self.critera = args

    def __str__(self):
        return "not any of: %s" % ", ".join([str(x) for x in self.critera])

    def get_matching_atoms(self, atoms, geometry):
        """returns List(Atom) that do not match any of the critera"""
        unmatching_atoms = []
        for criterion in self.critera:
            try:
                unmatch = geometry.find(criterion)
                unmatching_atoms.extend(unmatch)
            
            except LookupError:
                pass

        return [atom for atom in atoms if atom not in set(unmatching_atoms)]


class AnyTransitionMetal(Finder):
    """any atoms that are transition metals"""
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "any transition metal"

    def get_matching_atoms(self, atoms, geometry=None):
        """returns List(Atom) of atoms that are metals"""
        from AaronTools.const import TMETAL
        return [atom for atom in atoms if atom.element in TMETAL]


class AnyNonTransitionMetal(NotAny):
    """any atoms that are not transition metals"""
    def __init__(self):
        super().__init__(AnyTransitionMetal())

    def __str__(self):
        return "any non-transition metal"


class HasAttribute(Finder):
    """all atoms with the specified attribute"""
    def __init__(self, attribute):
        super().__init__()

        self.attribute_name = attribute

    def __str__(self):
        return "atoms with the '%s' attribute" % self.attribute_name

    def get_matching_atoms(self, atoms, geometry=None):
        """returns List(Atom) of atoms that have the attribute"""
        return [atom for atom in atoms if hasattr(atom, self.attribute_name)]



