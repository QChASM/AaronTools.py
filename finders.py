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

    def get_matching_atoms(self, atoms, geometry):
        """returns List(Atom) that are a certain number of bonds away from the given atom"""
        matching_atoms = []
        for atom in atoms:
            path = geometry.shortest_path(atom, self.central_atom)
            if len(path) - 1 == self.n_bonds:
                matching_atoms.append(atom)

        return matching_atoms


class WithinRadius(Finder):
    """within a specified radius of a point"""
    def __init__(self, point, radius):
        super().__init__()

        self.point = np.array(point)
        self.radius = radius

    def get_matching_atoms(self, atoms, geometry=None):
        """returns list(Atom) that are within a radius of a point"""
        matching_atoms = []
        for atom in atoms:
            coords = atom.coords
            d = np.linalg.norm(coords - self.point)
            if d < self.radius:
                matching_atoms.append(atom)

        return matching_atoms
            
