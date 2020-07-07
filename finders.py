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

    def __repr__(self):
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

    def __repr__(self):
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

    def __repr__(self):
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
    
    def __repr__(self):
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
    
    def __repr__(self):
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

    def __repr__(self):
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

    def __repr__(self):
        return "any transition metal"

    def get_matching_atoms(self, atoms, geometry=None):
        """returns List(Atom) of atoms that are metals"""
        from AaronTools.const import TMETAL
        return [atom for atom in atoms if atom.element in TMETAL]


class AnyNonTransitionMetal(NotAny):
    """any atoms that are not transition metals"""
    def __init__(self):
        super().__init__(AnyTransitionMetal())

    def __repr__(self):
        return "any non-transition metal"


class HasAttribute(Finder):
    """all atoms with the specified attribute"""
    def __init__(self, attribute):
        super().__init__()

        self.attribute_name = attribute

    def __repr__(self):
        return "atoms with the '%s' attribute" % self.attribute_name

    def get_matching_atoms(self, atoms, geometry=None):
        """returns List(Atom) of atoms that have the attribute"""
        return [atom for atom in atoms if hasattr(atom, self.attribute_name)]


class VSEPR(Finder):
    """atoms with the specified VSEPR geometry
    see Atom.get_shape for a list of valid vsepr_geometry strings"""
    def __init__(self, vsepr_geometry):
        super().__init__()
        
        self.vsepr = vsepr_geometry
    
    def __repr__(self):
        return "atoms with %s shape" % self.vsepr

    def get_matching_atoms(self, atoms, geometry=None):
        matching_atoms = []
        for atom in atoms:
            shape, score = atom.get_vsepr()
            if shape == self.vsepr and score < 0.5:
                matching_atoms.append(atom)
        
        return matching_atoms


class BondedElements(Finder):
    """atoms bonded to the specified neighboring elements
    if match_exact=True (default), elements must match exactly 
    e.g. BondedElements('C') will find
    atoms bonded to only one carbon and nothing else"""
    def __init__(self, *args, match_exact=True):
        super().__init__()
        
        self.elements = list(args)
        self.match_exact = match_exact
        
    def __repr__(self):
        if len(self.elements) == 0:
            return "atoms bonded to nothing"
        elif len(self.elements) == 1:
            return "atoms bonded to %s" % self.elements[0]
        else:
            return "atoms bonded to %s and %s" % (", ".join(self.elements[:-1]), self.elements[-1])
    
    def get_matching_atoms(self, atoms, geometry=None):
        matching_atoms = []
        if self.match_exact:
            ref = "".join(sorted(self.elements))
        else:
            ref = self.elements
        
        for atom in atoms:
            if self.match_exact:
                ele_list = [a.element for a in [ele for ele in atom.connected]]
                test = "".join(sorted(ele_list))
                if ref == test:
                    matching_atoms.append(atom)
            
            else:
                bonded_eles = [bonded_atom.element for bonded_atom in atom.connected]
                if all([ele in bonded_eles for ele in self.elements]):
                    matching_atoms.append(atom)
        
        return matching_atoms


class NumberOfBonds(Finder):
    """atoms with the specified number of bonds"""
    def __init__(self, num_bonds):
        super().__init__()
        
        self.num_bonds = num_bonds
    
    def __repr__(self):
        return "atoms with %i bonds" % self.num_bonds
    
    def get_matching_atoms(self, atoms, geometry=None):
        return [atom for atom in atoms if len(atom.connected) == self.num_bonds]
